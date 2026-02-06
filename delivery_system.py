"""
Mystery Delivery System - FastBox Logistics Simulator
=====================================================
This module simulates a day of delivery operations for FastBox.
It assigns packages to nearest agents, simulates deliveries,
and generates a performance report.

Author: FastBox Development Team
"""

import json
import math
from typing import Dict, List, Tuple, Any


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Package:
    """Represents a package to be delivered."""
    def __init__(self, package_id: str, warehouse_id: str, destination: Tuple[float, float]):
        self.id = package_id
        self.warehouse_id = warehouse_id
        self.destination = destination
    
    def __repr__(self):
        return f"Package(id={self.id}, warehouse={self.warehouse_id}, destination={self.destination})"


class Agent:
    """Represents a delivery agent."""
    def __init__(self, agent_id: str, location: Tuple[float, float]):
        self.id = agent_id
        self.location = location
        self.packages: List[Package] = []
        self.total_distance = 0.0
    
    def __repr__(self):
        return f"Agent(id={self.id}, location={self.location}, packages={len(self.packages)})"


class Warehouse:
    """Represents a warehouse."""
    def __init__(self, warehouse_id: str, location: Tuple[float, float]):
        self.id = warehouse_id
        self.location = location
    
    def __repr__(self):
        return f"Warehouse(id={self.id}, location={self.location})"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple
    
    Returns:
        Euclidean distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JSON data to handle both format variations.
    
    Format 1 (base_case.json):
        warehouses: [{"id": "W1", "location": [0, 0]}, ...]
        agents: [{"id": "A1", "location": [5, 5]}, ...]
        packages: [{"id": "P1", "warehouse_id": "W1", "destination": [30, 40]}, ...]
    
    Format 2 (test_case_*.json):
        warehouses: {"W1": [34, 29], ...}
        agents: {"A1": [89, 16], ...}
        packages: [{"id": "P1", "warehouse": "W5", "destination": [12, 7]}, ...]
    
    Args:
        data: Raw JSON data dictionary
    
    Returns:
        Normalized data dictionary with consistent structure
    """
    normalized = {}
    
    # Normalize warehouses
    if isinstance(data["warehouses"], list):
        # Format 1: List of objects with "id" and "location"
        normalized["warehouses"] = {
            w["id"]: tuple(w["location"]) for w in data["warehouses"]
        }
    else:
        # Format 2: Dictionary with id as key and location as value
        normalized["warehouses"] = {
            k: tuple(v) for k, v in data["warehouses"].items()
        }
    
    # Normalize agents
    if isinstance(data["agents"], list):
        # Format 1: List of objects with "id" and "location"
        normalized["agents"] = {
            a["id"]: tuple(a["location"]) for a in data["agents"]
        }
    else:
        # Format 2: Dictionary with id as key and location as value
        normalized["agents"] = {
            k: tuple(v) for k, v in data["agents"].items()
        }
    
    # Normalize packages - handle both "warehouse_id" and "warehouse" keys
    normalized["packages"] = []
    for pkg in data["packages"]:
        package_data = {
            "id": pkg["id"],
            "destination": tuple(pkg["destination"]),
            "warehouse": pkg.get("warehouse_id") or pkg.get("warehouse")  # Handle both formats
        }
        normalized["packages"].append(package_data)
    
    return normalized


# ============================================================================
# CORE LOGIC
# ============================================================================

def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load and parse JSON data from file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Parsed JSON data as dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def find_nearest_agent(package: Dict, agents: Dict[str, Tuple[float, float]], 
                       warehouses: Dict[str, Tuple[float, float]]) -> str:
    """
    Find the nearest agent to a package's warehouse.
    
    Args:
        package: Package data with warehouse location
        agents: Dictionary of agent_id -> location
        warehouses: Dictionary of warehouse_id -> location
    
    Returns:
        ID of the nearest agent
    """
    warehouse_location = warehouses[package["warehouse"]]
    
    min_distance = float('inf')
    nearest_agent_id = None
    
    for agent_id, agent_location in agents.items():
        distance = calculate_distance(agent_location, warehouse_location)
        if distance < min_distance:
            min_distance = distance
            nearest_agent_id = agent_id
    
    return nearest_agent_id


def simulate_delivery(agents: Dict[str, Tuple[float, float]], 
                      warehouses: Dict[str, Tuple[float, float]],
                      packages: List[Dict]) -> Dict[str, Dict]:
    """
    Simulate the delivery process.
    
    For each package:
    1. Find the nearest agent
    2. Calculate route: agent -> warehouse -> destination
    3. Add distance to agent's total
    
    Args:
        agents: Dictionary of agent_id -> location
        warehouses: Dictionary of warehouse_id -> location
        packages: List of package data
    
    Returns:
        Dictionary with agent statistics
    """
    # Initialize agent assignments and tracking
    agent_packages = {agent_id: [] for agent_id in agents}
    agent_distances = {agent_id: 0.0 for agent_id in agents}
    
    # Assign each package to the nearest agent
    for package in packages:
        nearest_agent = find_nearest_agent(package, agents, warehouses)
        agent_packages[nearest_agent].append(package)
    
    # Calculate delivery distances for each agent
    for agent_id in agents:
        agent_location = agents[agent_id]
        
        for package in agent_packages[agent_id]:
            warehouse_location = warehouses[package["warehouse"]]
            destination = package["destination"]
            
            # Route: agent -> warehouse -> destination
            distance_to_warehouse = calculate_distance(agent_location, warehouse_location)
            distance_warehouse_to_dest = calculate_distance(warehouse_location, destination)
            
            total_distance = distance_to_warehouse + distance_warehouse_to_dest
            agent_distances[agent_id] += total_distance
    
    # Build report
    report = {}
    for agent_id in agents:
        packages_delivered = len(agent_packages[agent_id])
        total_distance = agent_distances[agent_id]
        efficiency = total_distance / packages_delivered if packages_delivered > 0 else 0.0
        
        report[agent_id] = {
            "packages_delivered": packages_delivered,
            "total_distance": round(total_distance, 2),
            "efficiency": round(efficiency, 2)
        }
    
    return report


def find_best_agent(report: Dict[str, Dict]) -> str:
    """
    Find the most efficient agent (lowest efficiency value).
    Agents with 0 packages delivered are excluded from consideration.
    
    Args:
        report: Agent performance report
    
    Returns:
        ID of the most efficient agent
    """
    best_agent = None
    best_efficiency = float('inf')
    
    for agent_id, stats in report.items():
        if agent_id != "best_agent":
            # Exclude agents with 0 packages delivered
            if stats["packages_delivered"] > 0:
                if stats["efficiency"] < best_efficiency:
                    best_efficiency = stats["efficiency"]
                    best_agent = agent_id
    
    # If all agents delivered 0 packages, return None or first agent
    return best_agent


def generate_report(agents: Dict[str, Tuple[float, float]], 
                    warehouses: Dict[str, Tuple[float, float]],
                    packages: List[Dict]) -> Dict[str, Any]:
    """
    Generate the final delivery report.
    
    Args:
        agents: Dictionary of agent_id -> location
        warehouses: Dictionary of warehouse_id -> location
        packages: List of package data
    
    Returns:
        Complete report dictionary
    """
    # Simulate deliveries and get agent statistics
    agent_report = simulate_delivery(agents, warehouses, packages)
    
    # Find the most efficient agent
    best_agent = find_best_agent(agent_report)
    
    # Add best_agent to report
    agent_report["best_agent"] = best_agent
    
    return agent_report


def save_report(report: Dict, output_path: str = "report.json") -> None:
    """
    Save the report to a JSON file.
    
    Args:
        report: Report dictionary to save
        output_path: Path for the output file
    """
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the delivery simulation.
    """
    print("=" * 60)
    print("FastBox Delivery System Simulation")
    print("=" * 60)
    
    # Load data from JSON file
    input_file = "base_case.json"  # Change to test different cases
    print(f"\nLoading data from {input_file}...")
    
    try:
        raw_data = load_data(input_file)
        print(f"✓ Loaded {len(raw_data['packages'])} packages")
        print(f"✓ Loaded {len(raw_data['agents'])} agents")
        print(f"✓ Loaded {len(raw_data['warehouses'])} warehouses")
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        return
    
    # Normalize data to handle both JSON formats
    print("\nNormalizing data structure...")
    data = normalize_data(raw_data)
    print(f"✓ Normalized {len(data['warehouses'])} warehouses")
    print(f"✓ Normalized {len(data['agents'])} agents")
    print(f"✓ Normalized {len(data['packages'])} packages")
    
    # Generate report
    print("\nSimulating deliveries...")
    report = generate_report(data["agents"], data["warehouses"], data["packages"])
    
    # Display results
    print("\n" + "=" * 60)
    print("DELIVERY REPORT")
    print("=" * 60)
    
    for agent_id in sorted(report.keys()):
        if agent_id != "best_agent":
            stats = report[agent_id]
            print(f"\nAgent {agent_id}:")
            print(f"  Packages Delivered: {stats['packages_delivered']}")
            print(f"  Total Distance: {stats['total_distance']}")
            print(f"  Efficiency: {stats['efficiency']}")
    
    print(f"\n{'=' * 60}")
    print(f"BEST AGENT: {report['best_agent']}")
    print(f"{'=' * 60}")
    
    # Save report
    save_report(report)
    
    return report


# ============================================================================
# BONUS FEATURES (Optional Extensions)
# ============================================================================

class DeliverySimulatorWithBonus:
    """
    Extended delivery simulator with bonus features:
    - Random delivery delays
    - ASCII route visualization
    - Dynamic agent joining
    - CSV export of top performer
    """
    
    def __init__(self, agents: Dict, warehouses: Dict, packages: List):
        self.agents = agents
        self.warehouses = warehouses
        self.packages = packages
        self.delivery_times = {}
        self.route_logs = []
        
        # Run the simulation
        self.report = generate_report(agents, warehouses, packages)
        self.agent_packages = self._assign_packages()
        self._calculate_distances()
    
    def _assign_packages(self) -> Dict[str, List]:
        """Assign packages to agents."""
        agent_packages = {agent_id: [] for agent_id in self.agents}
        for package in self.packages:
            nearest_agent = find_nearest_agent(package, self.agents, self.warehouses)
            agent_packages[nearest_agent].append(package)
        return agent_packages
    
    def _calculate_distances(self) -> None:
        """Calculate distances for each agent."""
        self.agent_distances = {agent_id: 0.0 for agent_id in self.agents}
        for agent_id in self.agents:
            agent_location = self.agents[agent_id]
            for package in self.agent_packages[agent_id]:
                warehouse_location = self.warehouses[package["warehouse"]]
                destination = package["destination"]
                distance_to_warehouse = calculate_distance(agent_location, warehouse_location)
                distance_warehouse_to_dest = calculate_distance(warehouse_location, destination)
                self.agent_distances[agent_id] += distance_to_warehouse + distance_warehouse_to_dest
    
    def add_random_delays(self, delay_probability: float = 0.1, 
                          max_delay: float = 5.0) -> None:
        """
        Add random delays to deliveries.
        
        Args:
            delay_probability: Probability of a delay occurring
            max_delay: Maximum delay in minutes
        """
        import random
        for package in self.packages:
            if random.random() < delay_probability:
                delay = random.uniform(0, max_delay)
                self.delivery_times[package["id"]] = delay
    
    def visualize_route(self, agent_id: str) -> str:
        """
        Generate ASCII visualization of an agent's route.
        
        Args:
            agent_id: ID of the agent to visualize
        
        Returns:
            ASCII art representation of the route
        """
        if agent_id not in self.agent_packages or not self.agent_packages[agent_id]:
            return f"No packages assigned to Agent {agent_id}"
        
        agent_location = self.agents[agent_id]
        visualization = [
            f"\nRoute for Agent {agent_id}:",
            f"{'=' * 40}",
            f"Start: ({agent_location[0]:5.1f}, {agent_location[1]:5.1f})"
        ]
        
        for i, package in enumerate(self.agent_packages[agent_id], 1):
            warehouse = self.warehouses[package["warehouse"]]
            destination = package["destination"]
            visualization.append(
                f"  Stop {i}: Warehouse {package['warehouse']} "
                f"({warehouse[0]:5.1f}, {warehouse[1]:5.1f})"
            )
            visualization.append(
                f"         Destination {package['id']} "
                f"({destination[0]:5.1f}, {destination[1]:5.1f})"
            )
        
        visualization.append(f"{'=' * 40}")
        return "\n".join(visualization)
    
    def export_top_performer_csv(self, output_path: str = "top_performer.csv") -> None:
        """
        Export top performer's statistics to CSV.
        
        Args:
            output_path: Path for the CSV file
        """
        import csv
        
        best_agent_id = self.report["best_agent"]
        best_agent_stats = self.report[best_agent_id]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Agent ID", "Packages Delivered", "Total Distance", "Efficiency"])
            writer.writerow([
                best_agent_id,
                best_agent_stats["packages_delivered"],
                best_agent_stats["total_distance"],
                best_agent_stats["efficiency"]
            ])
        
        print(f"Top performer exported to {output_path}")
    
    def add_agent_midday(self, agent_id: str, location: Tuple[float, float]) -> None:
        """
        Add a new agent mid-day.
        
        Args:
            agent_id: ID of the new agent
            location: Starting location of the new agent
        """
        self.agents[agent_id] = location
        # Assign any unassigned packages to the new agent
        unassigned = []
        for package in self.packages:
            nearest = find_nearest_agent(package, self.agents, self.warehouses)
            if nearest == agent_id:
                unassigned.append(package)
        self.agent_packages[agent_id] = unassigned
        print(f"New agent {agent_id} joined at location {location}")


# ============================================================================
# TEST CASES
# ============================================================================

def run_test_cases():
    """
    Run all test cases and display results.
    """
    test_cases = [
        "base_case.json",
        "Python Assignment(Delivery System Test Cases)/test_case_1.json",
        "Python Assignment(Delivery System Test Cases)/test_case_2.json",
        "Python Assignment(Delivery System Test Cases)/test_case_3.json",
        "Python Assignment(Delivery System Test Cases)/test_case_4.json",
    ]
    
    print("RUNNING TEST CASES")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case}")
        print("-" * 40)
        
        try:
            raw_data = load_data(test_case)
            data = normalize_data(raw_data)
            report = generate_report(data["agents"], data["warehouses"], data["packages"])
            
            print(f"Packages: {len(data['packages'])}, Agents: {len(data['agents'])}")
            print(f"Best Agent: {report['best_agent']}")
            
            total_delivered = sum(
                report[agent]["packages_delivered"] 
                for agent in report if agent != "best_agent"
            )
            print(f"Total Delivered: {total_delivered}")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run the main simulation
    main()
    
    # Uncomment to run test cases:
    # run_test_cases()
