from typing import Dict, List, Optional
import re
from flask import session
import logging

logger = logging.getLogger(__name__)

def check_user_authorization() -> tuple[bool, Optional[str]]:
    """Check if the user is authorized"""
    try:
        if "username" not in session:
            logger.warning("Unauthorized access attempt - no username in session")
            return False, "You must be logged in to use this service."
        return True, None
    except Exception as e:
        logger.error(f"Error checking user authorization: {str(e)}")
        return False, "Error checking authorization. Please try again."

def check_medical_content(text: str) -> tuple[bool, Optional[str]]:
    """Check if the content is medically relevant"""
    try:
        medical_terms = [
            "patient", "diagnosis", "treatment", "symptoms", "disease",
            "medical", "clinical", "health", "doctor", "hospital",
            "medicine", "drug", "therapy", "procedure", "test"
        ]
        
        text_lower = text.lower()
        has_medical_terms = any(term in text_lower for term in medical_terms)
        
        if not has_medical_terms:
            logger.info("Non-medical content detected")
            return False, "The query does not appear to be related to medical content."
        return True, None
    except Exception as e:
        logger.error(f"Error checking medical content: {str(e)}")
        return False, "Error checking content. Please try again."

def check_personal_advice(text: str) -> tuple[bool, Optional[str]]:
    """Check if the content requests personal medical advice"""
    try:
        personal_advice_phrases = [
            "do i have", "diagnose my", "what disease do i have",
            "am i sick", "do i need surgery", "what medication should i take",
            "prescribe me", "what's wrong with me", "treat my",
            "my symptoms", "should i take", "do i need medication"
        ]
        
        text_lower = text.lower()
        for phrase in personal_advice_phrases:
            if phrase in text_lower:
                logger.info("Personal medical advice request detected")
                return False, "I cannot provide personal medical advice. Please consult with a healthcare professional."
        return True, None
    except Exception as e:
        logger.error(f"Error checking personal advice: {str(e)}")
        return False, "Error checking content. Please try again."

def check_illegal_activities(text: str) -> tuple[bool, Optional[str]]:
    """Check if the content requests illegal activities"""
    try:
        illegal_phrases = [
            "illegal drugs", "illegal substances", "make drugs",
            "home surgery", "avoid doctor", "self diagnose",
            "alternative to vaccines", "replace medication",
            "methamphetamine", "narcotics", "homemade drugs"
        ]
        
        text_lower = text.lower()
        for phrase in illegal_phrases:
            if phrase in text_lower:
                logger.warning("Illegal activity request detected")
                return False, "I cannot provide information about illegal activities."
        return True, None
    except Exception as e:
        logger.error(f"Error checking illegal activities: {str(e)}")
        return False, "Error checking content. Please try again."

def get_flows() -> Dict:
    """Return available flows for guardrails"""
    return {
        "check_user_authorization": check_user_authorization,
        "check_medical_content": check_medical_content,
        "check_personal_advice": check_personal_advice,
        "check_illegal_activities": check_illegal_activities
    } 