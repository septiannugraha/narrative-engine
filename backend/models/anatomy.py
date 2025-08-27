"""
Anatomical Model - Detailed body part tracking for consistent physical descriptions
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class BodySide(Enum):
    """Laterality of body parts"""
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    BOTH = "both"


@dataclass
class BodyPart:
    """Individual body part with specific features"""
    name: str
    side: Optional[BodySide] = None
    scars: List[str] = field(default_factory=list)
    tattoos: List[str] = field(default_factory=list)
    piercings: List[str] = field(default_factory=list)
    birthmarks: List[str] = field(default_factory=list)
    temporary_marks: List[str] = field(default_factory=list)  # bruises, hickeys, etc.
    
    def has_features(self) -> bool:
        """Check if this body part has any notable features"""
        return bool(self.scars or self.tattoos or self.piercings or 
                   self.birthmarks or self.temporary_marks)
    
    def get_description(self) -> str:
        """Get formatted description of this body part's features"""
        features = []
        if self.scars:
            features.extend([f"scar: {s}" for s in self.scars])
        if self.tattoos:
            features.extend([f"tattoo: {t}" for t in self.tattoos])
        if self.piercings:
            features.extend([f"piercing: {p}" for p in self.piercings])
        if self.birthmarks:
            features.extend([f"birthmark: {b}" for b in self.birthmarks])
        if self.temporary_marks:
            features.extend([f"mark: {m}" for m in self.temporary_marks])
        
        if features:
            part_name = f"{self.side.value} {self.name}" if self.side else self.name
            return f"{part_name}: {', '.join(features)}"
        return ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "side": self.side.value if self.side else None,
            "scars": self.scars,
            "tattoos": self.tattoos,
            "piercings": self.piercings,
            "birthmarks": self.birthmarks,
            "temporary_marks": self.temporary_marks
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BodyPart':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            side=BodySide(data["side"]) if data.get("side") else None,
            scars=data.get("scars", []),
            tattoos=data.get("tattoos", []),
            piercings=data.get("piercings", []),
            birthmarks=data.get("birthmarks", []),
            temporary_marks=data.get("temporary_marks", [])
        )


@dataclass
class AnatomicalModel:
    """Complete anatomical representation with all body parts"""
    
    # Head & Face
    forehead: BodyPart = field(default_factory=lambda: BodyPart("forehead", BodySide.CENTER))
    left_eyebrow: BodyPart = field(default_factory=lambda: BodyPart("eyebrow", BodySide.LEFT))
    right_eyebrow: BodyPart = field(default_factory=lambda: BodyPart("eyebrow", BodySide.RIGHT))
    left_eye: BodyPart = field(default_factory=lambda: BodyPart("eye", BodySide.LEFT))
    right_eye: BodyPart = field(default_factory=lambda: BodyPart("eye", BodySide.RIGHT))
    nose: BodyPart = field(default_factory=lambda: BodyPart("nose", BodySide.CENTER))
    left_cheek: BodyPart = field(default_factory=lambda: BodyPart("cheek", BodySide.LEFT))
    right_cheek: BodyPart = field(default_factory=lambda: BodyPart("cheek", BodySide.RIGHT))
    left_ear: BodyPart = field(default_factory=lambda: BodyPart("ear", BodySide.LEFT))
    right_ear: BodyPart = field(default_factory=lambda: BodyPart("ear", BodySide.RIGHT))
    upper_lip: BodyPart = field(default_factory=lambda: BodyPart("upper lip", BodySide.CENTER))
    lower_lip: BodyPart = field(default_factory=lambda: BodyPart("lower lip", BodySide.CENTER))
    chin: BodyPart = field(default_factory=lambda: BodyPart("chin", BodySide.CENTER))
    
    # Neck & Shoulders
    neck: BodyPart = field(default_factory=lambda: BodyPart("neck", BodySide.CENTER))
    throat: BodyPart = field(default_factory=lambda: BodyPart("throat", BodySide.CENTER))
    left_shoulder: BodyPart = field(default_factory=lambda: BodyPart("shoulder", BodySide.LEFT))
    right_shoulder: BodyPart = field(default_factory=lambda: BodyPart("shoulder", BodySide.RIGHT))
    
    # Arms
    left_upper_arm: BodyPart = field(default_factory=lambda: BodyPart("upper arm", BodySide.LEFT))
    right_upper_arm: BodyPart = field(default_factory=lambda: BodyPart("upper arm", BodySide.RIGHT))
    left_elbow: BodyPart = field(default_factory=lambda: BodyPart("elbow", BodySide.LEFT))
    right_elbow: BodyPart = field(default_factory=lambda: BodyPart("elbow", BodySide.RIGHT))
    left_forearm: BodyPart = field(default_factory=lambda: BodyPart("forearm", BodySide.LEFT))
    right_forearm: BodyPart = field(default_factory=lambda: BodyPart("forearm", BodySide.RIGHT))
    left_wrist: BodyPart = field(default_factory=lambda: BodyPart("wrist", BodySide.LEFT))
    right_wrist: BodyPart = field(default_factory=lambda: BodyPart("wrist", BodySide.RIGHT))
    left_hand: BodyPart = field(default_factory=lambda: BodyPart("hand", BodySide.LEFT))
    right_hand: BodyPart = field(default_factory=lambda: BodyPart("hand", BodySide.RIGHT))
    
    # Chest & Breasts
    chest: BodyPart = field(default_factory=lambda: BodyPart("chest", BodySide.CENTER))
    left_breast: BodyPart = field(default_factory=lambda: BodyPart("breast", BodySide.LEFT))
    right_breast: BodyPart = field(default_factory=lambda: BodyPart("breast", BodySide.RIGHT))
    left_nipple: BodyPart = field(default_factory=lambda: BodyPart("nipple", BodySide.LEFT))
    right_nipple: BodyPart = field(default_factory=lambda: BodyPart("nipple", BodySide.RIGHT))
    
    # Torso
    upper_back: BodyPart = field(default_factory=lambda: BodyPart("upper back", BodySide.CENTER))
    lower_back: BodyPart = field(default_factory=lambda: BodyPart("lower back", BodySide.CENTER))
    left_side: BodyPart = field(default_factory=lambda: BodyPart("side", BodySide.LEFT))
    right_side: BodyPart = field(default_factory=lambda: BodyPart("side", BodySide.RIGHT))
    abdomen: BodyPart = field(default_factory=lambda: BodyPart("abdomen", BodySide.CENTER))
    navel: BodyPart = field(default_factory=lambda: BodyPart("navel", BodySide.CENTER))
    
    # Hips & Bottom
    left_hip: BodyPart = field(default_factory=lambda: BodyPart("hip", BodySide.LEFT))
    right_hip: BodyPart = field(default_factory=lambda: BodyPart("hip", BodySide.RIGHT))
    left_buttock: BodyPart = field(default_factory=lambda: BodyPart("buttock", BodySide.LEFT))
    right_buttock: BodyPart = field(default_factory=lambda: BodyPart("buttock", BodySide.RIGHT))
    
    # Intimate Areas
    pubic_area: BodyPart = field(default_factory=lambda: BodyPart("pubic area", BodySide.CENTER))
    left_inner_thigh: BodyPart = field(default_factory=lambda: BodyPart("inner thigh", BodySide.LEFT))
    right_inner_thigh: BodyPart = field(default_factory=lambda: BodyPart("inner thigh", BodySide.RIGHT))
    
    # Legs
    left_thigh: BodyPart = field(default_factory=lambda: BodyPart("thigh", BodySide.LEFT))
    right_thigh: BodyPart = field(default_factory=lambda: BodyPart("thigh", BodySide.RIGHT))
    left_knee: BodyPart = field(default_factory=lambda: BodyPart("knee", BodySide.LEFT))
    right_knee: BodyPart = field(default_factory=lambda: BodyPart("knee", BodySide.RIGHT))
    left_calf: BodyPart = field(default_factory=lambda: BodyPart("calf", BodySide.LEFT))
    right_calf: BodyPart = field(default_factory=lambda: BodyPart("calf", BodySide.RIGHT))
    left_ankle: BodyPart = field(default_factory=lambda: BodyPart("ankle", BodySide.LEFT))
    right_ankle: BodyPart = field(default_factory=lambda: BodyPart("ankle", BodySide.RIGHT))
    left_foot: BodyPart = field(default_factory=lambda: BodyPart("foot", BodySide.LEFT))
    right_foot: BodyPart = field(default_factory=lambda: BodyPart("foot", BodySide.RIGHT))
    
    def get_all_features(self) -> List[str]:
        """Get all notable features across the entire body"""
        features = []
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in ['get_all_features', 'get_part', 
                                                                    'add_marking', 'to_dict', 'from_dict']:
                body_part = getattr(self, attr_name)
                if isinstance(body_part, BodyPart) and body_part.has_features():
                    description = body_part.get_description()
                    if description:
                        features.append(description)
        return features
    
    def get_part(self, part_name: str, side: Optional[BodySide] = None) -> Optional[BodyPart]:
        """Get a specific body part by name and side"""
        if side:
            full_name = f"{side.value}_{part_name}".replace(" ", "_")
        else:
            full_name = part_name.replace(" ", "_")
        
        return getattr(self, full_name, None)
    
    def add_marking(self, part_name: str, side: Optional[BodySide], 
                    marking_type: str, description: str):
        """Add a marking to a specific body part"""
        part = self.get_part(part_name, side)
        if part:
            if marking_type == "scar":
                part.scars.append(description)
            elif marking_type == "tattoo":
                part.tattoos.append(description)
            elif marking_type == "piercing":
                part.piercings.append(description)
            elif marking_type == "birthmark":
                part.birthmarks.append(description)
            elif marking_type == "temporary":
                part.temporary_marks.append(description)
    
    def to_dict(self) -> Dict:
        """Convert entire anatomical model to dictionary"""
        result = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in ['get_all_features', 'get_part', 
                                                                    'add_marking', 'to_dict', 'from_dict']:
                body_part = getattr(self, attr_name)
                if isinstance(body_part, BodyPart):
                    result[attr_name] = body_part.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AnatomicalModel':
        """Create anatomical model from dictionary"""
        model = cls()
        for part_name, part_data in data.items():
            if hasattr(model, part_name):
                setattr(model, part_name, BodyPart.from_dict(part_data))
        return model


class PhysicalMeasurements:
    """Detailed physical measurements for a character"""
    
    def __init__(self):
        self.height = ""  # e.g., "5'6\""
        self.weight = ""  # e.g., "125 lbs"
        self.bust = ""    # e.g., "34C"
        self.waist = ""   # e.g., "26\""
        self.hips = ""    # e.g., "36\""
        self.shoe_size = ""
        self.blood_type = ""
        
        # Additional measurements
        self.neck = ""
        self.shoulders = ""
        self.inseam = ""
        self.arm_length = ""
        
    def get_three_sizes(self) -> str:
        """Get the classic three measurements"""
        if self.bust and self.waist and self.hips:
            return f"{self.bust}-{self.waist}-{self.hips}"
        return ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "height": self.height,
            "weight": self.weight,
            "bust": self.bust,
            "waist": self.waist,
            "hips": self.hips,
            "shoe_size": self.shoe_size,
            "blood_type": self.blood_type,
            "neck": self.neck,
            "shoulders": self.shoulders,
            "inseam": self.inseam,
            "arm_length": self.arm_length
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PhysicalMeasurements':
        """Create from dictionary"""
        measurements = cls()
        for key, value in data.items():
            if hasattr(measurements, key):
                setattr(measurements, key, value)
        return measurements