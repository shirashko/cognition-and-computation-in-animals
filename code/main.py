
import heapq
import random
import matplotlib.pyplot as plt
import numpy as np

QUALITY_ֹNECTAR = 1
REGULAR_NECTAR_QUALITY = 0.5
AVERAGE_NECTAR_QUANTITY_IN_FLOWER = 0.5
MAX_VAL = 1.0
MIN_VAL = 0.0
SOCIAL_THRESHOLD = 0.5


class Bee:
    """
        Class representing a Bee with different traits and behaviors.
        """

    def __init__(self, mean_exploration=0.5, mean_boldness=0.5, mean_sociability=0.5, std_dev=0.2):
        """
               Initialize a Bee instance with traits and nectar-related attributes.

               :param mean_exploration: Mean value for exploration trait.
               :param mean_boldness: Mean value for boldness trait.
               :param mean_sociability: Mean value for sociability trait.
               :param std_dev: Standard deviation for generating traits.
               """
        # exploration property is expressed by the quality of the nectar the bee finds - if it likes to explore new
        # places, it's more likely she will find a more quality nectar. In the same time, the chance the bee will
        # die in a session increase with exploratory, as the more exploratory the bee, the more it will decide to go
        # and search for food in unknown places, and the more chances the bee will be in danger, because the area is
        # unknown (encounter enemy for example).
        self.exploration = max(MIN_VAL, min(MAX_VAL, random.gauss(mean_exploration, std_dev)))  # generate random
        # number following a Gaussian distribution

        # boldness attribute is expressed by the time the bee spends outside the hive, and the chance it takes by keep
        # looking for better flowers and not stick with the one she founds. this will translate to the amount of nectar
        # the bee will collect
        self.boldness = max(MIN_VAL, min(MAX_VAL, random.gauss(mean_boldness, std_dev)))

        self.sociability = max(MIN_VAL, min(MAX_VAL, random.gauss(mean_sociability, std_dev)))
        self.nectar_quantity = 0
        self.nectar_quality = 0
        self.alive = True

    def __lt__(self, other):
        return self.nectar_quality < other.nectar_quality

    def forage(self):
        """
        Simulate a foraging action by the Bee, influencing nectar quantity and quality based on traits.

        :return: Tuple containing nectar quantity and nectar quality found during foraging.
        """
        # Calculate the chance bee will die when search for food
        # If a bee is bolder, there are greater chance it will survive danger situations, and
        # more the bee is exploratory, the more dangers she can be encountered with
        # if the bee is bolder, she will spend more time outside the hive, so the more danger she exposed to
        alive = random.choices([1, 0], [1-(self.exploration * self.boldness), self.exploration*self.boldness])[0]
        if not alive:  # 0 nectar quality means the bee died
            self.nectar_quantity, self.nectar_quality = 0, 0
            self.alive = False
            return 0, 0

        self.nectar_quality = random.choices([random.uniform(0.5, 1.0), random.uniform(0.0, 0.7)],
                                             [self.exploration, 1 - self.exploration])[0]

        # boldness influence on the quantity of the nectar. if the bee is bold, it will not stick
        # with the same flower when the nectar is starting to run out, and she will take the chance
        # to look for other flowers. But there is a chance she wouldn't find such so the quantity
        # she will bring be random. on the other hand, if a bee is not bold, it will stick the same
        # flower and bring small amount of nectar
        self.nectar_quantity = random.choices([random.uniform(AVERAGE_NECTAR_QUANTITY_IN_FLOWER,1.0), AVERAGE_NECTAR_QUANTITY_IN_FLOWER],
                                              [self.boldness, 1 - self.boldness])[0]

        return self.nectar_quantity, self.nectar_quality

    def forage_based_on_dance(self, flower_quality):
        """
        Simulate a foraging action by the Bee based on the dance of another bee, influencing nectar quantity and quality.

        :param flower_quality: Nectar quality based on another bee's dance.
        :return: Tuple containing nectar quantity and nectar quality found during foraging.
        """
        # Calculate the chance bee will die when search for food
        # If a bee is bolder, there are greater chance it will survive danger situations, and
        # more the bee is exploratory, the more dangers she can be encountered with
        alive = random.choices([1,0], [0.7, 0.3])[0]
        if not alive:  # 0 nectar quality means the bee died
            self.nectar_quantity, self.nectar_quality = 0, 0
            self.alive = False
            return 0, 0
        self.nectar_quality = flower_quality
        self.nectar_quantity = random.choices([random.uniform(AVERAGE_NECTAR_QUANTITY_IN_FLOWER,1.0), AVERAGE_NECTAR_QUANTITY_IN_FLOWER],
                                              [self.boldness, 1 - self.boldness])[0]
        return self.nectar_quantity, self.nectar_quality


def create_hive(num_bees, std_dev=0.25, mean_exploration=0.5, mean_boldness=0.5, mean_sociability=0.5):
    """
    Create a hive of bees with given traits and simulate their foraging behavior.

    :param num_bees: Number of bees in the hive.
    :param std_dev: Standard deviation for generating traits.
    :param mean_exploration: Mean value for exploration trait.
    :param mean_boldness: Mean value for boldness trait.
    :param mean_sociability: Mean value for sociability trait.
    :return: Tuple containing hive's average nectar quantity, nectar quality, number of surviving bees, and average traits.
    """
    bees = [Bee(mean_exploration=mean_exploration, mean_boldness=mean_boldness, mean_sociability=mean_sociability,
                std_dev=std_dev) for _ in range(num_bees)]
    hive_nectar_quantity, hive_nectar_quality = 0, 0
    surviving_bees = 0
    total_exploration, total_aggressiveness, total_sociability = 0, 0, 0

    # add to the list all the bees which went by their own to find a food source and not followed other bee dance.
    # assuming these bees will come back to the hive and dance, so all the bees which are still in the hive,
    # will decide how to follow this rule: if the bee is sociable, it will follow the bee that is dancing and found the
    # highest quality food, and that is enough sociable by itself (both bees need to be enough sociable). Otherwise,
    # the bee will search for food by its own.

    dancing_bees_heap = []  # Max heap for dancing bees based on nectar quality
    cur_quan, cur_qual = 0, 0
    not_foraged_yet = True
    # not forage at all. the results are a bit different in the scenario.
    for cur_bee in bees:
        if cur_bee.sociability > SOCIAL_THRESHOLD:  # follow other social bees that dance if there exists one
            for _, dancing_bee in dancing_bees_heap:
                if dancing_bee.sociability > SOCIAL_THRESHOLD:
                    not_foraged_yet = False
                    cur_quan, cur_qual = cur_bee.forage_based_on_dance(dancing_bee.nectar_quality)
                    break  # Only follow one dancing bee
        if not_foraged_yet:  # forage on her own
            cur_quan, cur_qual = cur_bee.forage()
            heapq.heappush(dancing_bees_heap, (-cur_qual, cur_bee))  # Negate for max heap

        if cur_bee.alive:
            surviving_bees += 1
            hive_nectar_quantity += cur_quan
            hive_nectar_quality += cur_qual

        total_exploration += cur_bee.exploration
        total_aggressiveness += cur_bee.boldness
        total_sociability += cur_bee.sociability

    avg_hive_nectar_quantity = hive_nectar_quantity / num_bees
    avg_hive_nectar_quality = hive_nectar_quality / num_bees
    avg_exploration = total_exploration / num_bees
    avg_aggressiveness = total_aggressiveness / num_bees
    avg_sociability = total_sociability / num_bees

    return (avg_hive_nectar_quantity, avg_hive_nectar_quality, surviving_bees, avg_exploration, avg_aggressiveness,
            avg_sociability)


def simulate_hive_model(num_bees, num_simulations):
    """
    Simulate a hive model for multiple simulations and plot the results.

    :param num_bees: Number of bees in the hive.
    :param num_simulations: Number of simulations to run.
    """
    # Lists to store simulation results for the desired metrics
    avg_quantity = []
    avg_quality = []
    avg_survival = []

    for _ in range(num_simulations):
        hive_results = create_hive(num_bees)
        hive_nectar_quantity, hive_nectar_quality, surviving_bees = hive_results[0], hive_results[1], hive_results[2]

        avg_quantity.append(hive_nectar_quantity)
        avg_quality.append(hive_nectar_quality)
        avg_survival.append(surviving_bees / num_bees)

    # Plot the results for the desired metrics
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].hist(avg_quantity, bins=20, color='purple', alpha=0.7)
    axs[0].set_title("Average Nectar Quantity Distribution")

    axs[1].hist(avg_quality, bins=20, color='red', alpha=0.7)
    axs[1].set_title("Average Nectar Quality Distribution")

    axs[2].hist(avg_survival, bins=20, color='gray', alpha=0.7)
    axs[2].set_title("Average Bee Survival Distribution")

    for ax in axs:
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    fig.suptitle("Evaluating Hive Metrics for Bees: Insights from a Gaussian Distribution (μ=0.5, σ=0.2)")
    plt.tight_layout()
    plt.show()


def simulate_hives(num_hives, num_bees_per_hive, std_dev_list):
    """
    Simulate multiple hives with different trait standard deviations and plot the results.

    :param num_hives: Number of hives to simulate.
    :param num_bees_per_hive: Number of bees per hive.
    :param std_dev_list: List of standard deviations for generating traits.
    :return: Dictionary containing simulation results for different aspects.
    """
    results = {
        'std_dev': [],
        'avg_quantity': [],
        'avg_quality': [],
        'avg_survival': [],
        'avg_exploration': [],
        'avg_boldness': [],
        'avg_sociability': []
    }

    for std_dev in std_dev_list:
        total_quantity, total_quality, total_survival, total_exploration, total_boldness, total_sociability = 0, 0, 0, 0, 0, 0
        # check for this std, for this heterogeneity/homogeneity how the hive function on average
        for _ in range(num_hives):
            hive_results = create_hive(num_bees_per_hive, std_dev)
            total_quantity += hive_results[0]
            total_quality += hive_results[1]
            total_survival += hive_results[2] / num_bees_per_hive
            total_exploration += hive_results[3]
            total_boldness += hive_results[4]
            total_sociability += hive_results[5]

        results['std_dev'].append(std_dev)
        results['avg_quantity'].append(total_quantity / num_hives)
        results['avg_quality'].append(total_quality / num_hives)
        results['avg_survival'].append(total_survival / num_hives)
        results['avg_exploration'].append(total_exploration / num_hives)
        results['avg_boldness'].append(total_boldness / num_hives)
        results['avg_sociability'].append(total_sociability / num_hives)

    return results


def plot_simulation_results(results):
    """
    Plot simulation results for different aspects.

    :param results: Dictionary containing simulation results.
    """
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))

    # Nectar Quantity
    axs[0].plot(results['std_dev'], np.array(results['avg_quantity']), label='Avg Nectar Quantity', color='blue')
    axs[0].set_title('Average Nectar Quantity')
    axs[0].set_xlabel('Std')
    axs[0].set_ylabel('Average Value')
    axs[0].legend()

    # Nectar Quality
    axs[1].plot(results['std_dev'], np.array(results['avg_quality']), label='Avg Nectar Quality', color='red')
    axs[1].set_title('Average Nectar Quality')
    axs[1].set_xlabel('Std')
    axs[1].set_ylabel('Average Value')
    axs[1].legend()

    # Survival rate
    axs[2].plot(results['std_dev'], np.array(results['avg_survival']), label='Avg Bee Survival', color='green')
    axs[2].set_title('Average Bee Survival Rate')
    axs[2].set_xlabel('Std')
    axs[2].set_ylabel('Average Survival Rate')
    axs[2].legend()

    # Traits
    axs[3].plot(results['std_dev'], results['avg_exploration'], label='Avg Exploration', color='purple')
    axs[3].plot(results['std_dev'], results['avg_boldness'], label='Avg Boldness', color='cyan')
    axs[3].plot(results['std_dev'], results['avg_sociability'], label='Avg Sociability', color='orange')
    axs[3].set_title('Traits as Function of Standard Deviation')
    axs[3].set_xlabel('Std')
    axs[3].set_ylabel('Average Trait Value')
    axs[3].legend()

    # Add overall figure title
    fig.suptitle("Exploring Homogeneity vs Heterogeneity: Impact of Different Standard Deviations", fontsize=16)

    # Adjust layout
    plt.tight_layout()

    plt.show()


def simulate_hives_with_traits(num_hives, num_bees_per_hive, trait_combinations):
    """
    Simulate hives with different trait combinations and return results.

    :param num_hives: Number of hives to simulate for each combination.
    :param num_bees_per_hive: Number of bees per hive.
    :param trait_combinations: List of trait combinations to test.
    :return: List of tuples containing trait combination, average nectar quantity, and average nectar quality.
    """
    results = []

    for traits in trait_combinations:
        total_quantity, total_quality = 0, 0
        for _ in range(num_hives):
            hive_results = create_hive(num_bees_per_hive, mean_exploration=traits[0],
                                       mean_boldness=traits[1], mean_sociability=traits[2], std_dev=0)
            total_quantity += hive_results[0]
            total_quality += hive_results[1]
        avg_quantity = total_quantity / num_hives
        avg_quality = total_quality / num_hives
        results.append((traits, avg_quantity, avg_quality))

    return results


def find_optimal_combination(results):
    """
    Find the optimal trait combinations based on simulation results.

    :param results: List of tuples containing trait combination, average nectar quantity, and average nectar quality.
    :return: List of tuples representing the optimal trait combinations.
    """

    # for quality
    max_quality = max(result[2] for result in results)
    optimal_combinations_qual = [result for result in results if result[2] == max_quality]

    # for quantity
    max_quantity = max(result[1] for result in results)
    optimal_combinations_quan = [result for result in results if result[1] == max_quantity]

    return optimal_combinations_qual, optimal_combinations_quan


def plot_optimal_combination(combinations_with_results, optimal_combinations_qual, optimal_combinations_quan):
    """
    Plot the optimal trait combinations and their impact on nectar quantity and quality.

    :param combinations_with_results: List of tuples containing trait combination, average nectar quantity, and average nectar quality.
    :param optimal_combinations_qual: List of tuples representing optimal trait combinations for nectar quality.
    :param optimal_combinations_quan: List of tuples representing optimal trait combinations for nectar quantity.
    """
    # Extract data for plotting
    trait_combinations = [combination[0] for combination in combinations_with_results]
    quantities = [combination[1] for combination in combinations_with_results]
    qualities = [combination[2] for combination in combinations_with_results]

    # Create a grid of trait values
    exploration_values = [trait[0] for trait in trait_combinations]
    boldness_values = [trait[1] for trait in trait_combinations]
    sociability_values = [trait[2] for trait in trait_combinations]

    # Plotting
    fig = plt.figure(figsize=(12, 8))

    # Add suptitle to the whole figure
    fig.suptitle('Exploring the Impact of Various Average Hive Personality Combinations on Hive Metrics', fontsize=16)

    # Plot for Nectar Quantity
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(exploration_values, boldness_values, sociability_values, c=quantities, cmap='viridis')
    ax1.set_title('Nectar Quantity')
    ax1.set_xlabel('Exploration')
    ax1.set_ylabel('Boldness')
    ax1.set_zlabel('Sociability', labelpad=10)  # Adjust labelpad to move the label closer
    fig.colorbar(sc1, ax=ax1, label='Nectar Quantity')  # Add colorbar for quantity

    # Plot for Nectar Quality
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(exploration_values, boldness_values, sociability_values, c=qualities, cmap='plasma')
    ax2.set_title('Nectar Quality')
    ax2.set_xlabel('Exploration')
    ax2.set_ylabel('Boldness')
    ax2.set_zlabel('Sociability', labelpad=10)  # Adjust labelpad to move the label closer
    fig.colorbar(sc2, ax=ax2, label='Nectar Quality')  # Add colorbar for quality

    # Adjust subplot layout
    plt.tight_layout()

    plt.show()


def main():
    """
    Main function to run the simulation and analysis.
    """
    simulate_hive_model(num_bees=1000, num_simulations=1000)
    # std=0.25, mean=0.5

    # Simulate hives with varying standard deviations and plot the results
    # for heterogeneity vs homogeneity test
    results = simulate_hives(num_hives=1000, num_bees_per_hive=1000, std_dev_list=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plot_simulation_results(results)

    # Define trait combinations to test
    trait_combinations = []
    for i in range(1, 10, 2):
        for j in range(1, 10, 2):
            for k in range(1, 10, 2):
                trait_combinations.append((i/10, j/10, k/10))

    # Run the simulation with different trait combinations
    results = simulate_hives_with_traits(num_hives=100, num_bees_per_hive=1000, trait_combinations=trait_combinations)

    # Find the optimal trait combinations
    optimal_combinations_qual, optimal_combinations_quan = find_optimal_combination(results)

    # Plot the results
    plot_optimal_combination(results, optimal_combinations_qual, optimal_combinations_quan)

    # Print the optimal trait combinations
    for combination in optimal_combinations_quan:
        print(
            f"Optimal Combination: Exploration={combination[0][0]}, Boldness={combination[0][1]}, Sociability={combination[0][2]}")
        print(f"Average Nectar Quantity: {combination[1]}")

    # Print the optimal trait combinations
    for combination in optimal_combinations_qual:
        print(
            f"Optimal Combination: Exploration={combination[0][0]}, Boldness={combination[0][1]}, Sociability={combination[0][2]}")
        print(f"Average Nectar Quality: {combination[2]}\n")


if __name__ == "__main__":
    for _ in range(10):
        main()

