INTENT_LABELS = [
    'book_appointment', 'cancel_appointment', 'change_date', 'change_doctor',
    'change_service', 'change_time', 'check_appointment_status', 'check_my_appointments',
    'clinic_hours', 'clinic_location', 'complaint', 'confirm_appointment', 'confirmation',
    'consultation_fee', 'denial', 'doctor_availability', 'doctor_specialization',
    'emergency_request', 'goodbye', 'greeting', 'help', 'out_of_domain',
    'register_new_patient', 'repeat', 'reschedule_appointment', 'services_available',
    'slot_answer', 'transfer_to_human', 'update_contact_info', 'verify_patient_identity'
]

INTENT_TO_ID = {intent: i for i, intent in enumerate(INTENT_LABELS)}
ID_TO_INTENT = {i: intent for i, intent in enumerate(INTENT_LABELS)}

NUM_INTENTS = len(INTENT_LABELS)

SLOT_TAGS = [
    'O',
    'B-DATE', 'I-DATE',
    'B-DOCTOR_NAME', 'I-DOCTOR_NAME',
    'B-PATIENT_NAME', 'I-PATIENT_NAME',
    'B-PHONE', 'I-PHONE',
    'B-SERVICE_TYPE', 'I-SERVICE_TYPE',
    'B-TIME', 'I-TIME'
]

TAG_TO_ID = {tag: i for i, tag in enumerate(SLOT_TAGS)}
ID_TO_TAG = {i: tag for i, tag in enumerate(SLOT_TAGS)}

NUM_SLOT_LABELS = len(SLOT_TAGS)
