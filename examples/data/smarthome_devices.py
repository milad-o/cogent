"""
SmartHome Device Models and Controllers.

This module defines the core device abstractions for the SmartHome platform.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class DeviceState(Enum):
    """Possible states for a device."""
    ONLINE = "online"
    OFFLINE = "offline"
    PAIRING = "pairing"
    ERROR = "error"
    UPDATING = "updating"


class DeviceType(Enum):
    """Categories of supported devices."""
    LIGHT = "light"
    SWITCH = "switch"
    THERMOSTAT = "thermostat"
    SENSOR = "sensor"
    LOCK = "lock"
    CAMERA = "camera"
    SPEAKER = "speaker"
    APPLIANCE = "appliance"


@dataclass
class DeviceCapability:
    """Represents a capability that a device supports."""
    name: str
    type: str  # "boolean", "number", "string", "enum"
    read_only: bool = False
    min_value: float | None = None
    max_value: float | None = None
    enum_values: list[str] = field(default_factory=list)


@dataclass
class Device:
    """Base device representation."""
    id: str
    name: str
    device_type: DeviceType
    manufacturer: str
    model: str
    firmware_version: str
    state: DeviceState = DeviceState.OFFLINE
    room: str | None = None
    capabilities: list[DeviceCapability] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    last_seen: datetime | None = None
    
    def is_online(self) -> bool:
        """Check if device is currently online."""
        return self.state == DeviceState.ONLINE
    
    def update_attribute(self, key: str, value: Any) -> None:
        """Update a device attribute."""
        self.attributes[key] = value
        self.last_seen = datetime.now()


class DeviceController(ABC):
    """Abstract base class for device controllers."""
    
    @abstractmethod
    async def connect(self, device: Device) -> bool:
        """Establish connection to the device."""
        pass
    
    @abstractmethod
    async def disconnect(self, device: Device) -> None:
        """Disconnect from the device."""
        pass
    
    @abstractmethod
    async def send_command(self, device: Device, command: str, params: dict) -> dict:
        """Send a command to the device."""
        pass
    
    @abstractmethod
    async def get_state(self, device: Device) -> dict:
        """Get current state of the device."""
        pass


class LightController(DeviceController):
    """Controller for smart lights."""
    
    async def connect(self, device: Device) -> bool:
        """Connect to a light device."""
        if device.device_type != DeviceType.LIGHT:
            raise ValueError(f"Expected LIGHT device, got {device.device_type}")
        device.state = DeviceState.ONLINE
        return True
    
    async def disconnect(self, device: Device) -> None:
        """Disconnect from a light device."""
        device.state = DeviceState.OFFLINE
    
    async def send_command(self, device: Device, command: str, params: dict) -> dict:
        """Send command to light (turn_on, turn_off, set_brightness, set_color)."""
        if command == "turn_on":
            device.attributes["power"] = True
            return {"success": True, "power": True}
        elif command == "turn_off":
            device.attributes["power"] = False
            return {"success": True, "power": False}
        elif command == "set_brightness":
            brightness = params.get("brightness", 100)
            brightness = max(0, min(100, brightness))
            device.attributes["brightness"] = brightness
            return {"success": True, "brightness": brightness}
        elif command == "set_color":
            color = params.get("color", "#FFFFFF")
            device.attributes["color"] = color
            return {"success": True, "color": color}
        else:
            return {"success": False, "error": f"Unknown command: {command}"}
    
    async def get_state(self, device: Device) -> dict:
        """Get current light state."""
        return {
            "power": device.attributes.get("power", False),
            "brightness": device.attributes.get("brightness", 100),
            "color": device.attributes.get("color", "#FFFFFF"),
        }


class ThermostatController(DeviceController):
    """Controller for smart thermostats."""
    
    MODES = ["heat", "cool", "auto", "off"]
    
    async def connect(self, device: Device) -> bool:
        """Connect to a thermostat device."""
        if device.device_type != DeviceType.THERMOSTAT:
            raise ValueError(f"Expected THERMOSTAT device, got {device.device_type}")
        device.state = DeviceState.ONLINE
        # Initialize default state
        device.attributes.setdefault("current_temp", 70.0)
        device.attributes.setdefault("target_temp", 72.0)
        device.attributes.setdefault("mode", "auto")
        device.attributes.setdefault("humidity", 45)
        return True
    
    async def disconnect(self, device: Device) -> None:
        """Disconnect from thermostat."""
        device.state = DeviceState.OFFLINE
    
    async def send_command(self, device: Device, command: str, params: dict) -> dict:
        """Send command to thermostat."""
        if command == "set_temperature":
            temp = params.get("temperature", 72.0)
            temp = max(50, min(90, temp))  # Clamp to safe range
            device.attributes["target_temp"] = temp
            return {"success": True, "target_temp": temp}
        elif command == "set_mode":
            mode = params.get("mode", "auto")
            if mode not in self.MODES:
                return {"success": False, "error": f"Invalid mode: {mode}"}
            device.attributes["mode"] = mode
            return {"success": True, "mode": mode}
        else:
            return {"success": False, "error": f"Unknown command: {command}"}
    
    async def get_state(self, device: Device) -> dict:
        """Get current thermostat state."""
        return {
            "current_temp": device.attributes.get("current_temp", 70.0),
            "target_temp": device.attributes.get("target_temp", 72.0),
            "mode": device.attributes.get("mode", "auto"),
            "humidity": device.attributes.get("humidity", 45),
        }


class SensorController(DeviceController):
    """Controller for various sensors (motion, door, temperature, etc.)."""
    
    async def connect(self, device: Device) -> bool:
        """Connect to a sensor device."""
        if device.device_type != DeviceType.SENSOR:
            raise ValueError(f"Expected SENSOR device, got {device.device_type}")
        device.state = DeviceState.ONLINE
        return True
    
    async def disconnect(self, device: Device) -> None:
        """Disconnect from sensor."""
        device.state = DeviceState.OFFLINE
    
    async def send_command(self, device: Device, command: str, params: dict) -> dict:
        """Sensors are typically read-only, but some support configuration."""
        if command == "set_sensitivity":
            sensitivity = params.get("sensitivity", "medium")
            device.attributes["sensitivity"] = sensitivity
            return {"success": True, "sensitivity": sensitivity}
        elif command == "calibrate":
            device.attributes["calibrated"] = True
            return {"success": True, "calibrated": True}
        else:
            return {"success": False, "error": "Sensors are read-only"}
    
    async def get_state(self, device: Device) -> dict:
        """Get current sensor readings."""
        sensor_type = device.attributes.get("sensor_type", "unknown")
        if sensor_type == "motion":
            return {"motion_detected": device.attributes.get("motion", False)}
        elif sensor_type == "door":
            return {"open": device.attributes.get("open", False)}
        elif sensor_type == "temperature":
            return {"temperature": device.attributes.get("temperature", 70.0)}
        elif sensor_type == "humidity":
            return {"humidity": device.attributes.get("humidity", 50)}
        else:
            return device.attributes


# Factory for creating appropriate controllers
CONTROLLER_MAP: dict[DeviceType, type[DeviceController]] = {
    DeviceType.LIGHT: LightController,
    DeviceType.THERMOSTAT: ThermostatController,
    DeviceType.SENSOR: SensorController,
}


def get_controller(device_type: DeviceType) -> DeviceController:
    """Get the appropriate controller for a device type."""
    controller_class = CONTROLLER_MAP.get(device_type)
    if controller_class is None:
        raise ValueError(f"No controller available for {device_type}")
    return controller_class()


# Example devices for testing
SAMPLE_DEVICES = [
    Device(
        id="light-001",
        name="Living Room Light",
        device_type=DeviceType.LIGHT,
        manufacturer="Philips",
        model="Hue White A19",
        firmware_version="1.93.7",
        room="Living Room",
        capabilities=[
            DeviceCapability("power", "boolean"),
            DeviceCapability("brightness", "number", min_value=0, max_value=100),
            DeviceCapability("color", "string"),
        ],
    ),
    Device(
        id="thermo-001",
        name="Main Thermostat",
        device_type=DeviceType.THERMOSTAT,
        manufacturer="Ecobee",
        model="SmartThermostat Premium",
        firmware_version="4.8.7.171",
        room="Hallway",
        capabilities=[
            DeviceCapability("current_temp", "number", read_only=True),
            DeviceCapability("target_temp", "number", min_value=50, max_value=90),
            DeviceCapability("mode", "enum", enum_values=["heat", "cool", "auto", "off"]),
        ],
    ),
    Device(
        id="sensor-001",
        name="Front Door Sensor",
        device_type=DeviceType.SENSOR,
        manufacturer="Samsung",
        model="SmartThings Multipurpose",
        firmware_version="2.3.1",
        room="Entryway",
        attributes={"sensor_type": "door"},
        capabilities=[
            DeviceCapability("open", "boolean", read_only=True),
            DeviceCapability("temperature", "number", read_only=True),
        ],
    ),
]
