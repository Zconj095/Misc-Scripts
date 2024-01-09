from dataclasses import dataclass


@dataclass
class StatsModifier:
    value: float
    type: int 
    stackable: bool
    order: int
    has_duration: bool
    time: float
    
    
@dataclass    
class StatsValue:
    base_value: float
    max_value: float = -1
    has_max_value: bool = False
    modifiers: List[StatsModifier] = field(default_factory=list)
    
    @property
    def modified_value(self) -> float:
        # Calculation logic
        pass

        
@dataclass        
class StatsProgression:
    current_xp: int
    current_level: int 
    max_xp: int
    max_level: int
    progression_curve: List[float]
    
    def add_xp(self, xp: int):
        self.current_xp += xp
        
        # Logic to check level up
        pass

from dataclasses import dataclass, field
from typing import List, Callable
import time

@dataclass
class TimedModifier:
    modifier: StatsModifier
    time_added: float
    
    
@dataclass
class StatsValue:

    base_value: float
    max_value: float = -1
    has_max_value: bool = False
    
    modifiers: List[StatsModifier] = field(default_factory=list)
    timed_modifiers: List[TimedModifier] = field(default_factory=list)
    
    on_value_modified: Callable[[], None] = None
    
    @property
    def modified_value(self) -> float:
        value = self.base_value
        
        # Calculation logic
        # Sort modifiers
        # Apply each modifier
        
        return value
        
    def add_modifier(self, modifier: StatsModifier):
        if modifier.has_duration:
            timed_modifier = TimedModifier(modifier, time.time())
            self.timed_modifiers.append(timed_modifier)
        else:
            self.modifiers.append(modifier)
            
        self.on_value_modified()
        
    
    def remove_modifier(self, modifier: StatsModifier):
        if modifier in self.modifiers:
            self.modifiers.remove(modifier)
            
        self.on_value_modified()

            
@dataclass
class StatsProgression:

    current_xp: int
    current_level: int
    
    max_xp: int 
    max_level: int
    
    progression_curve: List[float]
    
    on_level_up: Callable[[], None] = None
    on_xp_added: Callable[[], None] = None
    
    def add_xp(self, xp: int):
        self.current_xp += xp
        self.on_xp_added()
        
        # Logic to check level up
        if self.current_xp >= self.get_xp_for_level(self.current_level + 1):
            self.current_level += 1  
            self.on_level_up()

    def get_xp_for_level(self, level: int) -> int:
        # Custom curve logic
        return xp


from dataclasses import dataclass, field
from typing import List, Callable
from enum import Enum
import math

class ModifierType(Enum):
    FLAT = 1
    PERCENT_ADD = 2 
    PERCENT_MULT = 3
    
    
@dataclass    
class StatsModifier:
    value: float
    type: ModifierType 
    stackable: bool = False
    order: int = 0
    
    def __post_init__(self):
        if self.order == 0:
            self.order = self.type.value
            

            
@dataclass            
class StatsValue:

    base_value: float
    max_value: float = -1
    
    modifiers: List[StatsModifier] = field(default_factory=list)
    
    def calculate_final_value(self) -> float:
      
        final_value = self.base_value
        sum_percent_add = 0

        # Sort modifiers by order
        sorted_modifiers = sorted(self.modifiers, key=lambda mod: mod.order)
        
        for modifier in sorted_modifiers:
            if modifier.type == ModifierType.FLAT:
                final_value += modifier.value
                
            elif modifier.type == ModifierType.PERCENT_ADD:
                sum_percent_add += modifier.value
                
                # Apply after current modifier type
                if modifier.stackable: 
                    final_value *= 1 + sum_percent_add  
                    sum_percent_add = 0
                    
            elif modifier.type == ModifierType.PERCENT_MULT:
                final_value *= 1 + modifier.value
                
        # Apply max value
        if self.max_value > -1:
            final_value = min(final_value, self.max_value)

        return round(final_value, 4)

from dataclasses import dataclass, field
from typing import List, Callable
from enum import Enum
import math

@dataclass
class StatsProgression:

    current_xp: int = 0
    current_level: int = 0
    
    max_xp: int  
    max_level: int
    
    progression_curve: List[float] = field(default_factory=list)
    
    on_level_up: Callable[[], None] = None
    on_xp_added: Callable[[], None] = None

    
    def add_xp(self, xp: int):
        self.current_xp += xp
        
        self.on_xp_added()
        
        if self.current_xp >= self.get_xp_for_level(self.current_level + 1):
            
            self.current_level += 1
            
            if self.current_level > self.max_level:
                self.current_level = self.max_level
                
            self.on_level_up() 


    def get_xp_for_level(self, level: int) -> int:
    
        # Normalize input level
        normalized = (level - self.max_level) / self.max_level
        
        # Get progression curve value 
        value = self.progression_curve[math.floor(normalized * len(self.progression_curve))]
        
        xp = value * self.max_xp
        
        return math.floor(xp)

from dataclasses import dataclass, field
from typing import List, Callable
from enum import Enum
import math

# PyQT for editor UI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

@dataclass
class StatsValue:

    base_value: float  
    max_value: float = -1
    modifiers: List[StatsModifier] = field(default_factory=list)
    
    on_value_modified: Callable[[], None] = None

    def editor_ui(self):
        window = QWidget()
        
        base_value_input = QDoubleSpinBox() 
        base_value_input.setValue(self.base_value)
        
        max_value_input = QDoubleSpinBox()
        max_value_input.setValue(self.max_value)
        
        modifiers_list = QListWidget()
        for mod in self.modifiers:
            item = QListWidgetItem(str(mod))
            modifiers_list.addItem(item)
            
        apply_button = QPushButton("Apply")
            
        # Layout 
        layout = QVBoxLayout()
        layout.addWidget(base_value_input)
        layout.addWidget(max_value_input)
        layout.addWidget(modifiers_list)     
        layout.addWidget(apply_button)
        
        window.setLayout(layout)
        
        # Logic to sync dataclass with UI
        # Connect signals
        
        
@dataclass   
class StatsProgression:  

    current_level: int = 0
    current_xp: int = 0
    max_level: int = 10
    
    on_level_up = pyqtSignal(int) # PyQT signal
    
    def editor_ui(self):
       window = QWidget()
       
       # UI elements
       level_label = QLabel(f"Level: {self.current_level}")
       
       add_xp_button = QPushButton("Add XP")
       add_xp_button.clicked.connect(self.add_xp)
       
       # Connect signal
       self.on_level_up.connect(update_level_label)
       
       layout = QVBoxLayout()
       layout.addWidget(level_label)  
       layout.addWidget(add_xp_button)
       
       window.setLayout(layout)


import json
from dataclasses import dataclass, field, asdict
from typing import List

@dataclass 
class StatsModifier:
    value: float
    type: int
    
    def to_json(self):
        return json.dumps(asdict(self))
    
    @staticmethod
    def from_json(json_str):
        json_dict = json.loads(json_str)
        return StatsModifier(**json_dict)

        
@dataclass
class StatsValue:  
    base_value: float
    modifiers: List[StatsModifier] = field(default_factory=list)
    
    def to_json(self):
        return json.dumps({
            "base_value": self.base_value,
            "modifiers": [mod.to_json() for mod in self.modifiers]  
        })
       
    @staticmethod 
    def from_json(json_str):
        json_dict = json.loads(json_str)
        mods = [StatsModifier.from_json(mod) for mod in json_dict["modifiers"]] 
        return StatsValue(
            base_value=json_dict["base_value"],
            modifiers=mods
        )

        
@dataclass       
class StatsProgression:
    current_xp: int 
    current_level: int
    
    def to_json(self):
        return json.dumps(asdict(self))
    
    @staticmethod
    def from_json(json_str):
        json_dict = json.loads(json_str)
        return StatsProgression(**json_dict)


from dataclasses import dataclass
from typing import Callable, List

class StatsEvent:
    pass

class ValueModifiedEvent(StatsEvent):
    value: float

class Subscribers:
    def __init__(self):
        self.subscribers = []
        
    def subscribe(self, callback):
        self.subscribers.append(callback) 

    def notify(self, event):
        for sub in self.subscribers:
            sub(event)
            
            
@dataclass        
class StatsValue:
    base_value: float  
    modifiers: List[StatsModifier]
    
    on_value_modified = Subscribers()
    
    @property
    def modified_value(self) -> float:
        # Calculation logic
        
        self.on_value_modified.notify(
            ValueModifiedEvent(modified_value)  
        )
    
        
def handle_value_modified(event: ValueModifiedEvent):
    print(f"Value changed to {event.value}")
    
stats = StatsValue(5.0)
stats.on_value_modified.subscribe(handle_value_modified)

stats.base_value = 10.0 # Will notify subscribers

from dataclasses import dataclass
from typing import List, Union, Callable
from enum import Enum

class ModifierType(Enum):
    FLAT = 'flat'
    PERCENT_ADD = 'percent_add'
    
class InvalidModifier(Exception): 
    pass

@dataclass    
class StatsModifier:
    value: float
    type: ModifierType
    order: int = 0
    
    @classmethod
    def validate(cls, value):
        if not isinstance(value, float):
            raise InvalidModifier("Value must be float")
            
        if not isinstance(type, ModifierType):
            raise InvalidModifier("Invalid modifier type")
            

@dataclass
class StatsValue:
    base_value: float 
    modifiers: List[StatsModifier]
    max_value: Union[int, float] = -1

    @classmethod
    def validate(cls, modifiers):
        for mod in modifiers:
            StatsModifier.validate(mod)
            
def subscriber(event: Callable[[StatsEvent], None]):
    # Check callable
    pass
        
@dataclass  
class StatsProgression:
    xp_for_level: Callable[[int], int]
    current_level: int = 1

StatsProgression.validate(SomeProgression)


import unittest
from stats import StatsModifier, StatsValue, ModifierType


class TestStatsModifier(unittest.TestCase):

    def test_create_modifier(self):
        mod = StatsModifier(5, ModifierType.FLAT)
        self.assertEqual(mod.value, 5) 
        self.assertEqual(mod.type, ModifierType.FLAT)

    def test_invalid_modifier_value(self):
        with self.assertRaises(InvalidModifier):
            StatsModifier("invalid", ModifierType.FLAT)
            

class TestStatsValue(unittest.TestCase):

    def test_base_value(self):
        value = StatsValue(10)
        self.assertEqual(value.base_value, 10)
    
    def test_modified_value(self):
        mod1 = StatsModifier(5, ModifierType.FLAT)  
        mod2 = StatsModifier(0.5, ModifierType.PERCENT_ADD)
        
        value = StatsValue(10, [mod1, mod2])
        self.assertEqual(value.modified_value, 15)
        
        
    def test_max_value(self):
        mod = StatsModifier(5, ModifierType.FLAT)
        
        value = StatsValue(10, [mod], max_value=12) 
        self.assertEqual(value.modified_value, 12)


from dataclasses import dataclass
from typing import Callable
from enum import Enum

class ModifierType(Enum): 
    FORMULA = "formula"

@dataclass
class Formula:
    formula_text: str # Store as text
    
    # Compile to executable on load
    formula_exec: Callable[[float], float]  
    
    @classmethod
    def build(cls, formula_text):
        # Logic to compile string formula to executable
        
        return Formula(formula_text, compiled_formula)
    
    
@dataclass
class StatsModifier:
    value: float
    type: ModifierType
    
    # New attribute
    formula: Formula = None


@dataclass
class StatsValue:

    def calculate_final_value(self):
        base_val = self.base_value
        
        for modifier in self.modifiers:
            
            if modifier.type == ModifierType.FORMULA:
                
                # Execute formula 
                base_val = modifier.formula.formula_exec(base_val)
                
            else:
                # Regular logic
                
        return base_val
                
formula_text = "value * 1.5 + 10"
formula = Formula.build(formula_text)
mod = StatsModifier(0, ModifierType.FORMULA, formula=formula)




