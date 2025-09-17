from .exp_fleet_membership_validation import FleetMembershipValidationRunner
from .experiment2 import Experiment2Runner
from .experiment4 import Experiment4Runner
from .exp_runner_base import ExperimentRunner
from .experiment_service_catalog import ServiceCatalogValidationRunner
from .exp_event_time_validation_bkp import EventTimeRepairRunner
from .exp_event_window_identification import EventWindowIdentificationRunner
from .experiment_field_integrity_repair import FieldIntegrityRepairRunner
from .exp_single_shot_processing import SingleShotCleaning
from .exp_zero_shot import ZeroShotFunctionalCleaning
from .exp_one_shot import OneShotFunctionalCleaning

EXPERIMENT_RUNNERS = {
    'fleet_membership_validation': FleetMembershipValidationRunner,
    'experiment2': Experiment2Runner,
    'experiment4': Experiment4Runner,
    # Assuming this is a base runner for service_catalog validation
    'service_catalog_validation': ServiceCatalogValidationRunner,
    # Placeholder for event time repair runner
    'event_time_repair': EventTimeRepairRunner,
    # Assuming this is the same as EventWindowIdentificationRunner
    'event_window_identification': EventWindowIdentificationRunner,
    'field_integrity_repair': FieldIntegrityRepairRunner,
    'single_shot_cleaning': SingleShotCleaning,
    'zero_shot': ZeroShotFunctionalCleaning,
    "one_shot": OneShotFunctionalCleaning,
}
