# Physics MCP Server

An MCP (Model Context Protocol) server for introductory algebra-based physics that provides tools to help students solve problems and visualize physics concepts.

## Features

- **Kinematics Solver**: Solves 1D kinematics problems with constant acceleration
- **Projectile Motion Analyzer**: Calculates trajectories and key parameters for projectile motion
- **Force Calculator**: Implements Newton's laws of motion
- **Circuit Analyzer**: Analyzes simple DC circuits with resistors in series and parallel
- **Energy Calculator**: Calculates potential and kinetic energy

## Installation

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -e .
   ```

## Usage

You can run the server directly:

```
python physics.py
```

Or via the MCP CLI:

```
mcp run physics
```

## Tools

### solve_kinematics

Solves a 1D kinematics problem with constant acceleration, providing at least 3 of 5 parameters:
- Initial velocity
- Final velocity
- Time
- Displacement
- Acceleration

### analyze_projectile_motion

Analyzes projectile motion given:
- Initial velocity
- Launch angle
- Initial height (optional)

### calculate_force

Calculates force using Newton's Second Law (F = ma).

### analyze_electric_circuit

Analyzes a simple DC circuit with resistors in series and/or parallel.

### calculate_energy

Calculates potential and kinetic energy for an object.

## Prompting Examples

Below are examples of how to prompt each tool in the physics MCP server. These examples can be used by students to solve physics problems and visualize concepts.

### Kinematics Example Prompts

```
# Example 1: Find acceleration and displacement
solve_kinematics initial_velocity=5 final_velocity=15 time=2.5

# Example 2: Find time and final velocity
solve_kinematics initial_velocity=0 acceleration=9.8 displacement=20

# Example 3: Calculate time for a car to stop
solve_kinematics initial_velocity=25 final_velocity=0 acceleration=-5
```

### Projectile Motion Example Prompts

```
# Example 1: Analyze a ball thrown at an angle
analyze_projectile_motion initial_velocity=20 angle_degrees=45

# Example 2: Analyze a projectile launched from a height
analyze_projectile_motion initial_velocity=15 angle_degrees=30 height=10

# Example 3: Analyze a horizontal launch
analyze_projectile_motion initial_velocity=10 angle_degrees=0 height=50
```

### Force Calculation Example Prompts

```
# Example 1: Calculate force needed to accelerate a car
calculate_force mass=1500 acceleration=2

# Example 2: Find force required to lift an object
calculate_force mass=5 acceleration=9.8

# Example 3: Calculate force for a lab experiment
calculate_force mass=0.25 acceleration=4.5
```

### Circuit Analysis Example Prompts

```
# Example 1: Analyze a series circuit
analyze_electric_circuit voltage=12 resistors=[{"value": 100, "type": "series"}, {"value": 200, "type": "series"}]

# Example 2: Analyze a parallel circuit
analyze_electric_circuit voltage=9 resistors=[{"value": 50, "type": "parallel", "group_id": "1"}, {"value": 100, "type": "parallel", "group_id": "1"}]

# Example 3: Analyze a mixed circuit (series and parallel)
analyze_electric_circuit voltage=24 resistors=[{"value": 100, "type": "series"}, {"value": 200, "type": "parallel", "group_id": "1"}, {"value": 300, "type": "parallel", "group_id": "1"}]
```

### Energy Calculation Example Prompts

```
# Example 1: Calculate both potential and kinetic energy
calculate_energy mass=2 height=10 velocity=5

# Example 2: Calculate only potential energy
calculate_energy mass=5 height=20

# Example 3: Calculate only kinetic energy
calculate_energy mass=0.5 velocity=30
```

## Integrating with AI Assistant Workflows

For instructors and students using AI assistants, you can integrate this MCP server into your workflows by asking the AI to use specific tools.

Example prompts for AI assistants:

1. "Can you use the `analyze_projectile_motion` tool to help me understand the trajectory of a baseball thrown at 20 m/s at a 35-degree angle?"

2. "I'm working on a circuit problem. Please use `analyze_electric_circuit` to solve for a 12V circuit with a 100-ohm resistor in series and two 200-ohm resistors in parallel."

3. "Can you show me the acceleration of a car that increases its speed from 0 to 60 mph in 8 seconds using the `solve_kinematics` tool?"

These prompts will direct the AI to use the appropriate physics tools from the MCP server to help solve problems and provide visual explanations.
