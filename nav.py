from datetime import datetime
from typing import Union
from math import sqrt, cos, sin, pi, isnan

from numpy import ndarray, array, zeros, eye, isnan as npisnan
from scipy.signal import butter
from ahrs.filters import EKF, Madgwick

# from geopy.distance import geodesic

from livefilter import PureLiveSosFilter, LiveMeanFilter, PureTripleLiveOneSectionSosFilter
from pyCRGI.pure import get_value
from madgwick.madgwickFast import updateMARGFast


GRAVITY = 9.80665  # m / s ** 2
AIR_DENSITY = 1.225  # density at STP in kg / m ** 3
DRAG_CONVERSION = 0.07803855  # frontal area in drag is in units of 0.84 ft ** 2 - this converts to m ** 2
DEFAULT_DRAG = AIR_DENSITY * 0.004834396004686504 * 0.5 * DRAG_CONVERSION  # Mean drag coefficient over all cars
DEG_TO_RAD = pi / 180
MPS_TO_KMH = 3.6
GAS_TO_EMISSIONS = 2.31  # 2.31 kg of CO2 produced for every litre of gasoline burned
EMISSIONS_TO_CO2 = 0.95  # proportion of CO2 produced
EMISSIONS_TO_CO = 0.01  # proportion of CO produced
EMISSIONS_TO_NOX = 0.021  # proportion of nitrous oxides produced
EMISSIONS_TO_PARTICULATE = 0.0005  # proportion of particulate matter produced
EMISSIONS_TO_HC = 0.019  # proportion of unburned hydrocarbons produced

EPS = 1e-12
GPS_REFRESH_TIME = 1.2  # seconds

# Data obtained from https://doi.org/10.3390/en6010117, tables 5, 6
REGRESSION_COEFFICIENTS = (
    (0.81, 4.82),
    (0.72, 4.97),
    (0.59, 5.46),
    (0.71, 5.77),
    (1.62, 3.21),
    (1.40, 3.87),
    (1.22, 4.20),
    (0.92, 7.47),
    (1.54, 8.78),
    (1.63, 4.53),
    (1.41, 4.67),
    (1.11, 7.22),
    (1.56, 9.29),
    (2.10, 7.86),
    (1.95, 6.30),
    (1.62, 8.21),
)


class Nav:
    '''
    Class to update navigation parameters from IMU, magnetometer, and GPS measurements and compute fuel economy.
    '''

    def __init__(
        self,
        smoothing_critical_freq: float = 0.03,
        vz_depth: int = 3,
        period: float = 0.01,
        algo: str = 'madgwick',
        displacement: Union[float, None] = None,
        is_supercharged: Union[float, None] = None,
        drag_coeff: Union[float, None] = None,
        smooth_fc: bool = True,
        fc_smoothing_critical_freq: float = 0.02,
        imu_damping: float = 0.1,
        fc_reduction_factor: float = 0.5,
    ) -> None:
        '''
        Args:
            smoothing_critical_freq: float - critical frequency of the butterworth filter applied to accel
                measurements
            vz_depth: int - number of previous GPS iterations to consider when computing vertical velocity
            period: float - IMU update period
            algo: str - AHRS algorithm. One of 'EKF', 'Madgwick'
            displacement: float | None - Engine displacement in L. Can be set later with set_vehicle_params
            is_supercahrged: bool | None - Whether or not the vehicle is supercharged or turbocharged.
                Can be set later with set_vehicle_params
            drag_coef: float | None - Drag coefficient times frontal area divided by vehicle mass.
                Can be set later with set_vehicle_params. The units for this are [0.84 ft ** 2 / kg]
            smooth_fc: bool - whether or not to smooth the outgoing fuel consumption values
            fc_smoothing_critical_freq: float - critical frequency of the butterworth filter applied to fuel
                consumption
            imu_damping: float - factor to damp the IMU measurements as GPS are more trustworthy
            fc_reduction_factor: float - reduce emissions by this factor for accuracy
        '''
        # Initialize navigation parameters
        self.v = zeros(3)  # Velocity in the vehicle frame
        self.a = zeros(3)  # Acceleration in the vehicle frame (sans gravity)
        self.v_llf = zeros(3)  # Velocity in the local level frame
        self.heading = None  # Heading of the car in degrees clockwise from North
        self.R_l2v = eye(3)
        # self.average_speed = LiveMeanFilter()

        # Initialize the filters for smoothing incomming accel and gyro data
        sos = butter(2, smoothing_critical_freq, output='sos', fs=None, btype='lowpass')[0]
        # self.accel_filter = PureTripleLiveOneSectionSosFilter(sos)
        self.accel_no_g_filter = PureTripleLiveOneSectionSosFilter(sos)
        # self.gyro_filter = PureTripleLiveOneSectionSosFilter(sos)

        # Initialize time-related IMU parameters
        self.t0 = None
        self.prev_timestamp = None
        self.period = period  # IMU update period
        self.latest_raw_imu = None

        # Initialize GPS-provided parameters
        self.prev_alt = None
        self.v_z_filter = LiveMeanFilter(vz_depth)
        self.prev_alt_timestamp = None
        # self.total_distance = 0
        self.prev_lat_long = None
        self.gps_time = 0

        # Initialize parameters for the AHRS algorithm
        self.ahrs = None  # AHRS algorithm
        self.Q_ahrs = None  # Quaternion in AHRS notation
        self.Q_s2l = None  # Quaternion in ENGO 623 notation
        # self.R_s2l = None  # Rotation matrix corresponding to self.Q_s2l

        # Set the AHRS algorithm
        if (algo := algo.lower()) not in (valid_algos := ('ekf', 'madgwick')):
            print(f'WARNING: algo must be one of {valid_algos}, not {algo}. Defaulting to Madgwick.')
            algo = 'madgwick'
        self.algo = algo
        self.algo_initialized = False

        # Initialize values for the fuel consumption computation
        if displacement is None and is_supercharged is None:
            self.edi = None  # Engine displacement index
        elif displacement is None or is_supercharged is None:
            print(
                'WARNING: Arguments displacement and is_supercharged must both be supplied or set later with set_vehicle_params'
            )
            self.edi = None
        else:
            self.edi = self._get_edi(displacement, is_supercharged)
        self.drag_coeff = DEFAULT_DRAG if drag_coeff is None else AIR_DENSITY * drag_coeff * 0.5 * DRAG_CONVERSION

        # Instantiate parameters to track fuel use
        # Update these every time we update the velocity
        self.current_fc = 0.0  # Current fuel consumption in mL / s
        self.total_fc = 0.0  # Total fuel consumption in L
        sos = butter(2, fc_smoothing_critical_freq, output='sos', fs=None, btype='lowpass')[0]
        self.fc_filter = PureLiveSosFilter(sos) if smooth_fc else None
        self.imu_damping = imu_damping
        self.fc_reduction_factor = fc_reduction_factor

    @staticmethod
    def normsq3(x):
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2]

    @staticmethod
    def _get_ref_field(lat: float, long: float, alt: float, return_inclination: bool = False) -> Union[ndarray, float]:
        '''
        Get the reference field from pyCRGI

        Args:
            lat: float - latitude in degrees
            long: float - longitude in degrees
            alt: float - altitude in meters
            return_inclination: bool - If true, return magnetic dip rather than reference field. Default: False

        Returns:
            reference field as a ndarray if return_inclination = False else magnetic dip as a float
        '''
        # Compute the year
        now = datetime.now().timetuple()
        year = now.tm_year + now.tm_yday / (365 if now.tm_year % 4 != 0 else 366)

        # Get the mag field from the IGRF-13 model
        mag_field = array(get_value(lat, long, alt, year))

        if return_inclination:
            return mag_field[1]

        # Convert from NED frame to ENU frame
        ref_field = mag_field[[3, 2, 4]]  # Reference field in nT
        ref_field[-1] = -ref_field[-1]
        return ref_field

    @staticmethod
    def _quaternion_to_matrix(q: ndarray) -> ndarray:
        '''
        Convert a normalized quaternion (in ENGO 623 notation) to a rotation matrix

        Args:
            q: ndarray - quaternion of shape (4,)

        Returns:
            ndarray of shape (3, 3)
        '''
        q1, q2, q3, q4 = q
        qq1, qq2, qq3, qq4 = q * q
        q1q2 = q1 * q2
        q3q4 = q3 * q4
        q1q3 = q1 * q3
        q2q4 = q2 * q4
        q1q4 = q1 * q4
        q2q3 = q2 * q3
        qq2mqq3 = qq2 - qq3
        qq4mqq1 = qq4 - qq1
        return array(
            [
                [qq1 - qq2 - qq3 + qq4, 2 * (q1q2 - q3q4), 2 * (q1q3 + q2q4)],
                [2 * (q1q2 + q3q4), qq4mqq1 + qq2mqq3, 2 * (q2q3 - q1q4)],
                [2 * (q1q3 - q2q4), 2 * (q2q3 + q1q4), qq4mqq1 - qq2mqq3],
            ]
        )

    @staticmethod
    def _rotate_with_quaternion(q: ndarray, v: ndarray) -> ndarray:
        '''
        Rotate using the quaternion
        Faster than using the matrix approach

        Args:
            q: ndarray - quaternion of shape (4,)
            v: ndarray - vector of shape (3,)

        Returns:
            ndarray - rotated vector of shape (3,)
        '''
        q0, q1, q2, q3 = q
        v0, v1, v2 = v

        a = 2.0 * (q0 * v0 + q1 * v1 + q2 * v2)
        b = q3 * q3 - q0 * q0 - q1 * q1 - q2 * q2
        c = 2.0 * q3

        return array(
            [
                a * q0 + b * v0 + c * (q1 * v2 - q2 * v1),
                a * q1 + b * v1 + c * (q2 * v0 - q0 * v2),
                a * q2 + b * v2 + c * (q0 * v1 - q1 * v0),
            ]
        )

    @staticmethod
    def _get_edi(displacement: float, is_supercharged: bool) -> int:
        '''
        Get the engine displacement index

        Args:
            displacement: float - engine displacement
            is_supercharged: bool - whether or not the vehicle is supercharged or turbocharged

        Returns:
            index: int
        '''
        if displacement < 1.6:
            edi = 0
        elif displacement < 2.5:
            edi = 1
        elif displacement < 4:
            edi = 2
        else:
            edi = 3
        if is_supercharged and edi != 3:
            edi += 1
        return edi

    @staticmethod
    def _get_fc_params(speed: float, edi: int):
        '''
        Get the regression coefficients for the linear fit of fuel consumption to VSP

        Args:
            speed: float - vehicle speed in m/s
            edi: int - engine displacement index

        Returns:
            tuple[float] - fuel consumption regression coefficients
        '''
        if speed < 5.556:  # 20 km/h
            speed_index = 0
        elif speed < 11.11:  # 40 km/h
            speed_index = 1
        elif speed < 16.67:  # 60 km/h
            speed_index = 2
        else:
            speed_index = 3
        return REGRESSION_COEFFICIENTS[4 * edi + speed_index]

    def process_imu_update(
        self, timestamp: float, accel: ndarray, accel_no_g: ndarray, gyro: ndarray, mag: ndarray
    ) -> None:
        '''
        Process a measurement from the IMU

        Args:
            timestamp: float
            accel: ndarray of shape (3,) - IMU acceleration in m / s ** 2 including gravity
            accel_no_g: ndarray of shape (3,) - IMU acceleration in m / s ** 2 excluding gravity
            gyro: ndarray of shape (3,) - IMU angular velocity in rad / s
            mag: ndarray of shape (3,) - Magnetic field in nT
        '''

        if any(
            x < EPS for x in (self.normsq3(accel), self.normsq3(accel_no_g), self.normsq3(gyro), self.normsq3(mag))
        ):
            print('WARNING: Received zero values in process_imu_update. Returning.')
            return

        assert not npisnan(accel).any(), 'ERROR: accel value is NaN in process_imu_update'
        assert not npisnan(accel_no_g).any(), 'ERROR: accel_no_gravity value is NaN in process_imu_update'
        assert not npisnan(gyro).any(), 'ERROR: gyro value is NaN in process_imu_update'
        assert not npisnan(mag).any(), 'ERROR: mag value is NaN in process_imu_update'

        if self.t0 is None:
            self.t0 = timestamp

        # Can't really do much on the first iteration since we don't know delta t
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return

        # Compute the time difference
        timediff = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp

        # Smooth the incomming accelerometer and gyro measurements
        # accel = array(self.accel_filter.process(accel))
        accel_no_g = array(self.accel_no_g_filter.process(accel_no_g))
        # gyro = array(self.gyro_filter.process(gyro))
        self.latest_raw_imu = [accel, gyro, mag]

        # If we have no reference field, we have no orientation
        if not self.algo_initialized:
            # print('WARNING: Cannot update navigation without orientation. GPS update required')
            return

        # Update the AHRS algo for orientation information
        if self.algo == 'madgwick':
            # self.Q_ahrs = self.ahrs.updateMARG(self.Q_ahrs, gyr=gyro, acc=accel, mag=mag)
            self.Q_ahrs = updateMARGFast(self.Q_ahrs, gyr=gyro, acc=accel, mag=mag, dt=timediff)
        elif self.algo == 'ekf':
            self.Q_ahrs = self.ahrs.update(self.Q_ahrs, gyr=gyro, acc=accel, mag=mag)
        assert not npisnan(self.Q_ahrs).any(), 'ERROR: self.Q_ahrs has NaN values in process_imu_update'
        self.Q_s2l = array([self.Q_ahrs[1], self.Q_ahrs[2], self.Q_ahrs[3], self.Q_ahrs[0]])

        # Compute the updated rotation matrix
        # self.R_s2l = self._quaternion_to_matrix(self.Q_s2l)

        # Rotate the acceleration (sans gravity) to the local level frame
        # accel_llf = self.R_s2l @ accel_no_g
        accel_llf = self._rotate_with_quaternion(self.Q_s2l, accel_no_g)  # Faster than rotating with the matrix
        accel_llf = accel_llf * self.imu_damping

        # Update the velocity
        self.v_llf += timediff * accel_llf

        # Use heading to rotate the local level frame velocity to the vehicle frame velocity
        # Heading vector in the LLF: [sin H, cos H, 0]
        # Need to rotate the y-axis of the local level frame to the align with the heading vector
        if self.heading is not None:
            sinh, cosh = sin(self.heading * DEG_TO_RAD), cos(self.heading * DEG_TO_RAD)
            self.R_l2v = array([[cosh, -sinh, 0], [sinh, cosh, 0], [0, 0, 1]])

        # Rotate the local level frame velocity and acceleration (sans gravity) to the vehicle frame
        self.v = (self.R_l2v @ self.v_llf.reshape((3, 1))).reshape(3)
        self.a = (self.R_l2v @ accel_llf.reshape((3, 1))).reshape(3)

        # Update the fuel consumption
        self._update_fuel_consumption(timediff)

        # # Update the average speed
        # self.average_speed.process(sqrt(self.v[1] ** 2 + self.v[2] ** 2))

    def process_gps_update(
        self, timestamp: float, lat: float, long: float, alt: float, heading: float, speed: float
    ) -> None:
        '''
        Process a measurement from the GPS

        Args:
            timestamp: float - time in s
            lat: float - latitude in degrees
            long: float - longitude in degrees
            alt: float - altitude in m
            heading: float - heading in degrees clockwise from North
            speed: float - speed in m / s

        Returns:
            None
        '''
        # Return if there is no GPS update and a second has not passed
        if self.prev_lat_long == (lat, long) and timestamp - self.gps_time < GPS_REFRESH_TIME:
            return
        self.gps_time = timestamp

        # Return if we can't compute rotation matrices
        if self.latest_raw_imu is None:
            return

        if not self.algo_initialized:
            self.algo_initialized = True

            # Initialize the AHRS algorithm
            if self.algo == 'ekf':
                # Get the reference field at the current location
                ref_field = self._get_ref_field(lat, long, alt)
                self.ahrs = EKF(
                    acc=self.latest_raw_imu[0].reshape((1, 3)),
                    gyr=self.latest_raw_imu[1].reshape((1, 3)),
                    mag=self.latest_raw_imu[2].reshape((1, 3)),
                    frequency=1 / self.period,
                    frame='ENU',
                    magnetic_ref=ref_field,
                )
            elif self.algo == 'madgwick':
                self.ahrs = Madgwick(
                    acc=self.latest_raw_imu[0].reshape((1, 3)),
                    gyr=self.latest_raw_imu[1].reshape((1, 3)),
                    mag=self.latest_raw_imu[2].reshape((1, 3)),
                    frequency=1 / self.period,
                )
            self.Q_ahrs = self.ahrs.Q[0]
            assert not npisnan(self.Q_ahrs).any(), 'ERROR: self.Q_ahrs has NaN values in process_gps_update'
            # Convert to ENGO 623 quaternion convention
            self.Q_s2l = array([self.Q_ahrs[1], self.Q_ahrs[2], self.Q_ahrs[3], self.Q_ahrs[0]])

            # Compute the rotation matrix from smartphone to ENU frame
            # self.R_s2l = self._quaternion_to_matrix(self.Q_s2l)

        # Update heading
        if heading is not None:
            self.heading = heading

        # Compute the vertical velocity and update the altitude and GPS timestamp
        if alt is not None:
            if self.prev_alt is None and alt is not None:
                v_z = 0
            else:
                v_z = self.v_z_filter.process((alt - self.prev_alt) / (timestamp - self.prev_alt_timestamp))
            self.prev_alt = alt
            self.prev_alt_timestamp = timestamp

            self.v[2] = v_z

        # Reset the speed
        if speed is not None:
            self.v[0] = 0
            self.v[1] = speed

        # Reset the local level frame velocity using the GPS measurements
        if alt is not None or speed is not None:
            self.v_llf = (self.R_l2v.T @ self.v.reshape((3, 1))).reshape(3)

        # # Update the total distance travelled
        # if self.prev_lat_long is None:
        #     self.prev_lat_long = (lat, long)
        # else:
        #     self.total_distance += geodesic((lat, long), self.prev_lat_long).meters
        #     self.prev_lat_long = (lat, long)

        self.prev_lat_long = (lat, long)

    def get_motion(self, speed_only: bool = True, max_digits: Union[int, None] = None, kmh: bool = False) -> dict:
        '''
        Get the current velocity and acceleration

        Args:
            speed_only: bool - If true, only return the speed of the vehicle
            max_digits: int | None - Maximum number of digits in the result
            kmh: bool - If true, return the result in km/h

        Returns:
            dict - most recent velocity and acceleration in the vehicle frame
        '''
        unit_conversion = MPS_TO_KMH if kmh else 1

        if speed_only:
            speed = sqrt(self.v[1] ** 2 + self.v[2] ** 2) * unit_conversion
            assert isinstance(speed, (float, int)) and not isnan(speed), 'ERROR: Speed is not a float'
            if max_digits is not None:
                return {'speed': float(str(speed)[: max_digits + 1])}
            return {'speed': speed}

        if max_digits is not None:
            return {
                'velocity': float(str(self.v * unit_conversion)[: max_digits + 1]),
                'acceleration': float(str(self.a * unit_conversion)[: max_digits + 1]),
            }
        return {'velocity': self.v * unit_conversion, 'acceleration': self.a * unit_conversion}

    def set_vehicle_params(
        self,
        displacement: Union[float, None] = None,
        is_supercharged: Union[bool, None] = None,
        drag_coeff: Union[float, None] = None,
        reset: bool = False,
    ) -> None:
        '''
        Set the vehicle parameters.  This must be done before calling get_fuel or get_emissions.

        Args:
            displacement: float | None - engine displacement
            is_supercharged: float | None - must be supplied at the same time as displacement
            drag_coeff: float | None - drag coefficient times frontal area divided by vehicle mass
            reset: bool - must be set to true to reset parameters that have already been set
        '''
        param_warning = lambda: print(
            'WARNING: Vehicle parameters already set. Use reset=True to reset the parameters.'
        )

        # Set the drag coefficient, if supplied
        if drag_coeff is not None:
            if self.drag_coeff != DEFAULT_DRAG and reset == False:
                param_warning()
            else:
                self.drag_coeff = AIR_DENSITY * drag_coeff * 0.5 * DRAG_CONVERSION

        # Set the engine displacement index if displacement and is_supercharged are both supplied
        if displacement is None and is_supercharged is None:
            return
        if displacement is None or is_supercharged is None:
            print('WARNING: Arguments displacement and is_supercharged must both be given.')
            return
        if self.edi is not None and reset == False:
            param_warning()
            return
        self.edi = self._get_edi(displacement, is_supercharged)

    def _update_fuel_consumption(self, timestep) -> None:
        '''
        Compute the instantaneous fuel consumption

        Args:
            timestep: float - the current timestep
        '''
        # Get the vehicle speed and regression parameters
        speed = sqrt(self.v[1] ** 2 + self.v[2] ** 2)
        a, b = self._get_fc_params(speed, self.edi)
        v_norm = sqrt(self.v[0] ** 2 + speed * speed)

        # Compute vehicle specific power in m ** 2 / s ** 3
        vsp = 1.1 * (self.v * self.a).sum() + GRAVITY * self.v[2] + self.drag_coeff * v_norm**3 + 0.132 * v_norm

        # Fuel consumption in mL / s and L
        self.current_fc = b if vsp <= 0 else a * vsp + b
        self.current_fc *= self.fc_reduction_factor
        if self.fc_filter is not None:
            self.current_fc = self.fc_filter.process(self.current_fc)
        self.total_fc += self.current_fc * timestep * 0.001  # Convert mL to L

    def get_fuel(self, return_totals: bool = False, max_digits: Union[int, None] = None) -> dict:
        '''
        Return the best estimates of the current and total fuel consumption

        Args:
            return_totals: bool - Whether or not to return total consumption information
            max_digits: int | None - Maximum number of digits in the result

        Returns:
            dict: Current fuel usage in mL / s and total fuel usage in L
        '''
        assert isinstance(self.current_fc, (float, int)) and not isnan(
            self.current_fc
        ), 'ERROR: Current FC is not an int or float'

        if return_totals:
            assert isinstance(self.total_fc, (float, int)) and not isnan(
                self.total_fc
            ), 'ERROR: Total FC is not an int or float'
            if max_digits is not None:
                return {
                    'fuel_current': float(str(self.current_fc)[: max_digits + 1]),
                    'fuel_total': float(str(self.total_fc)[: max_digits + 1]),
                }
            return {'fuel_current': self.current_fc, 'fuel_total': self.total_fc}
        if max_digits is not None:
            return {'fuel_current': float(str(self.current_fc)[: max_digits + 1])}
        return {'fuel_current': self.current_fc}

    def get_emissions(self, return_totals: bool = False, max_digits: Union[int, None] = None) -> dict:
        '''
        Return the best estimates of the current and cumulative emissions

        Args:
            return_totals: bool - Whether or not to return total emissions information
            max_digits: int | None - Maximum number of digits in the result

        Returns:
            dict: Current and total emissions information.
                Keys indicate the type of emission.
                Current usage in g / s.  Total consumption is in kg.

        '''
        if self.edi is None:
            print('WARNING: Vehicle parameters not set. Call set_vehicle_params to set.')
            return {}

        emissions_current = GAS_TO_EMISSIONS * self.current_fc
        co2_current = EMISSIONS_TO_CO2 * emissions_current
        co_current = EMISSIONS_TO_CO * emissions_current
        nox_current = EMISSIONS_TO_NOX * emissions_current
        particulate_current = EMISSIONS_TO_PARTICULATE * emissions_current
        hc_current = EMISSIONS_TO_HC * emissions_current

        if max_digits is None:
            emissions = {
                'co2_current': co2_current,
                'co_current': co_current,
                'nox_current': nox_current,
                'particulate_current': particulate_current,
                'unburned_hc_current': hc_current,
            }
        else:
            emissions = {
                'co2_current': float(str(co2_current)[: max_digits + 1]),
                'co_current': float(str(co_current)[: max_digits + 1]),
                'nox_current': float(str(nox_current)[: max_digits + 1]),
                'particulate_current': float(str(particulate_current)[: max_digits + 1]),
                'unburned_hc_current': float(str(hc_current)[: max_digits + 1]),
            }

        assert all(
            isinstance(x, (float, int)) and not isnan(x) for x in emissions.values()
        ), 'ERROR: Emissions computed as null'

        if not return_totals:
            return emissions

        emissions_total = GAS_TO_EMISSIONS * self.total_fc
        co2_total = EMISSIONS_TO_CO2 * emissions_total
        co_total = EMISSIONS_TO_CO * emissions_total
        nox_total = EMISSIONS_TO_NOX * emissions_total
        particulate_total = EMISSIONS_TO_PARTICULATE * emissions_total
        hc_total = EMISSIONS_TO_HC * emissions_total

        if max_digits is None:
            totals = {
                'co2_total': co2_total,
                'co_total': co_total,
                'nox_total': nox_total,
                'particulate_total': particulate_total,
                'unburned_hc_total': hc_total,
            }
        else:
            totals = {
                'co2_total': float(str(co2_total)[: max_digits + 1]),
                'co_total': float(str(co_total)[: max_digits + 1]),
                'nox_total': float(str(nox_total)[: max_digits + 1]),
                'particulate_total': float(str(particulate_total)[: max_digits + 1]),
                'unburned_hc_total': float(str(hc_total)[: max_digits + 1]),
            }

        return {
            **emissions,
            **totals,
        }

    def get_trip_metrics(self, max_digits: Union[int, None] = None):
        '''
        Compute aggregate metrics about the current trip

        Returns:
            tuple[float] - total distance travelled in meters and elapsed time in seconds
        '''
        if self.prev_timestamp is None:
            elapsed_time = 0
        else:
            elapsed_time = self.prev_timestamp + self.period - self.t0

        # return self.total_distance, elapsed_time
        if max_digits is not None:
            return float(str(elapsed_time)[: max_digits + 1])
        return elapsed_time
