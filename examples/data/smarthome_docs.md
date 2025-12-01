# SmartHome IoT Platform Documentation

## Overview

SmartHome is a comprehensive IoT platform for managing connected devices in residential and commercial buildings. The platform supports over 200 device types from 50+ manufacturers.

## Architecture

### Core Components

1. **Device Gateway**: Handles communication with physical devices via Zigbee, Z-Wave, WiFi, and Bluetooth protocols.
2. **Event Bus**: Real-time message broker using Redis Streams for device events.
3. **Automation Engine**: Rule-based automation with support for complex triggers and conditions.
4. **API Server**: RESTful API for third-party integrations.

### Data Flow

```
Device → Gateway → Event Bus → Automation Engine → Actions
                            ↓
                      Time Series DB
```

## Device Management

### Supported Protocols

| Protocol | Range | Power | Best For |
|----------|-------|-------|----------|
| Zigbee 3.0 | 10-20m | Low | Sensors, lights |
| Z-Wave Plus | 30-100m | Low | Locks, thermostats |
| WiFi | Varies | High | Cameras, speakers |
| Bluetooth LE | 10m | Very Low | Wearables, beacons |

### Adding a New Device

1. Put the device in pairing mode
2. Navigate to Settings → Devices → Add Device
3. Select the protocol type
4. Follow the on-screen instructions
5. Assign the device to a room

## Automation Rules

### Trigger Types

- **Time-based**: Schedule actions at specific times or intervals
- **Device state**: React to device state changes
- **Location**: Trigger based on user presence (geofencing)
- **Weather**: Integrate with weather APIs for weather-based automation

### Example Rule: Morning Routine

```yaml
name: "Morning Routine"
trigger:
  type: time
  at: "06:30"
  days: [mon, tue, wed, thu, fri]
conditions:
  - entity: person.john
    state: home
actions:
  - service: light.turn_on
    target: living_room
    brightness: 50%
  - service: thermostat.set
    target: main_floor
    temperature: 72
  - service: coffee_maker.brew
    target: kitchen
```

## Energy Management

The platform includes comprehensive energy monitoring:

- Real-time power consumption per device
- Historical usage graphs and trends
- Cost estimation based on utility rates
- Peak usage alerts and recommendations
- Solar panel integration support

### Energy Saving Tips

1. Use motion sensors to auto-off lights in unoccupied rooms
2. Set thermostats to eco mode when away
3. Schedule high-power appliances during off-peak hours
4. Monitor standby power consumption

## Security Features

### Access Control

- Multi-factor authentication required for all users
- Role-based permissions (Admin, User, Guest)
- API keys with granular scope control
- Audit logs for all configuration changes

### Encryption

- All device communication encrypted with AES-256
- TLS 1.3 for API connections
- End-to-end encryption for camera streams
- Secure key storage using hardware security modules

## API Reference

### Authentication

All API requests require a Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.smarthome.io/v2/devices
```

### Rate Limits

- Standard tier: 100 requests/minute
- Pro tier: 1000 requests/minute
- Enterprise: Unlimited

### Common Endpoints

- `GET /devices` - List all devices
- `GET /devices/{id}` - Get device details
- `POST /devices/{id}/command` - Send command to device
- `GET /automations` - List automation rules
- `POST /automations` - Create new automation

## Troubleshooting

### Device Offline

1. Check device power supply
2. Verify network connectivity
3. Check distance from gateway
4. Try removing and re-pairing the device

### High Latency

1. Reduce number of devices per gateway
2. Check for wireless interference
3. Update device firmware
4. Consider adding mesh repeaters

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.2.0 | 2025-11-15 | Added Matter protocol support |
| 3.1.0 | 2025-09-01 | Energy dashboard redesign |
| 3.0.0 | 2025-06-01 | New automation engine |
| 2.5.0 | 2025-03-01 | Camera AI features |
