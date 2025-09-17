# Labels and constants for the Fleet Membership Validation experiment

# Fleet Membership Validation

LABEL_MAIN_RECORD_CLEAN = "none"
LABEL_MAIN_RECORD_VEHICLE_IN_FLEET = "vehicle_in_fleet"
LABEL_MAIN_RECORD_VEHICLE_NOT_IN_FLEET = "vehicle_not_in_fleet"
LABEL_MAIN_RECORD_WRONG_VEHICLE_ID = "wrong_vehicle_id"
LABEL_MAIN_RECORD_WRONG_END_DATE = "wrong_end_date"
LABEL_MAIN_RECORD_TEST = "digital_system_test"
# Ontology Validation

LABEL_MAIN_RECORD_INVALID_COMPONENT = "invalid_component"
LABEL_MAIN_RECORD_MISMATCHED_HIERARCHY = "mismatched_hierarchy"
LABEL_MAIN_RECORD_ONTOLOGY_FIELD_SWAP = "ontology_field_swap"
LABEL_MAIN_RECORD_BLANK_ONTOLOGY_FIELD = "blank_ontology_field"

# Device Tab Labels

LABEL_DEVICE_TAB_NONE = "none"
LABEL_DEVICE_TAB_CORRUPTED_VIN = "corrupted_vin"
LABEL_DEVICE_TAB_NOISY_NAME = "noisy_name_field"
LABEL_DEVICE_TAB_UNSTANDARDIZED_NAME = "unstandardized_name_field"
LABEL_DEVICE_GENERAL_NOISE = "general_noise"

# Activity Dates Repair

LABEL_MAIN_RECORD_CORRECT_DATES = "both_dates_correct"
LABEL_MAIN_RECORD_WRONG_END_DATE = "wrong_end_date"
LABEL_MAIN_RECORD_WRONG_START_DATE = "wrong_start_date"
LABEL_MAIN_RECORD_INCORRECT_DATES = "both_dates_incorrect"

# Noise Types
LABEL_NOISE_TYPE_NONE = "none"
LABEL_NOISE_TYPE_GENERATIVE = "generative"
LABEL_NOISE_TYPE_CORRUPTIVE = "corruptive"

# Field Integrity Repair

LABEL_FIELD_INTEGRITY_MISSING_VALUE = "missing_value"
LABEL_FIELD_INTEGRITY_ALREADY_CLEAN = "none"
LABEL_FIELD_INTEGRITY_CATEGORICAL_TYPO = "categorical_typo"

# Experiments Id

EXP_TYPE_FLEET_MEMBERSHIP_VALIDATION = "fleet_membership_validation"
EXP_TYPE_SERVICE_CATALOG_VALIDATION = "service_catalog_validation"
EXP_TYPE_EVENT_TIME_REPAIR = "event_time_repair"
EXP_TYPE_EVENT_WINDOW_IDENTIFICATION = "event_window_identification"
EXP_TYPE_FIELD_INTEGRITY_REPAIR = "field_integrity_repair"
EXP_TYPE_SINGLE_SHOT_CLEANING = "single_shot_cleaning"
EXP_TYPE_SINGLE_SHOT_FUNCTIONAL = "single_shot_functional_cleaning"

# Classification task

LABEL_CLASS_ACCEPT = "accept"
LABEL_CLASS_REJECT = "reject"
LABEL_CLASS_UPDATE = "update"
LABEL_CLASS_FAILED = "failed"
