import random

class DecodingPolicyState:
    def __init__(self, num_blocks=4, steps=128, gen_length=128, possible_temperatures=[0.0, 0.1, 0.2], possible_remasking_strategies=["low_confidence", "random"]):  
        assert gen_length > num_blocks, "gen_length must be greater than num_blocks."
        assert steps >= gen_length, "steps must be greater than or equal to gen_length."   
        assert steps % num_blocks == 0, "steps must be divisible by num_blocks."   
        self.num_blocks = num_blocks
        self.steps = steps
        self.gen_length = gen_length

        self.possible_temperatures = possible_temperatures
        self.possible_remasking_strategies = possible_remasking_strategies

        self.temperature_schedule = []
        self.remasking_strategy_schedule = []
        self.block_schedule = []
        self.extra_step_proportions = []

    def initialize_default_policy(self):
        block_length = self.gen_length // self.num_blocks
        remainder = self.gen_length % self.num_blocks

        lowest_possible_temperature = min(self.possible_temperatures)
        base_remasking_strategy = "low_confidence" if "low_confidence" in self.possible_remasking_strategies else self.possible_remasking_strategies[0]
        for i in range(self.num_blocks):
            self.temperature_schedule.append(lowest_possible_temperature)
            self.remasking_strategy_schedule.append(base_remasking_strategy)
            self.block_schedule.append(block_length + (1 if i == 0 else 0) if i < remainder else block_length)
            self.extra_step_proportions.append(round(1.0 / self.num_blocks, 2))

    def clone(self):
        new = DecodingPolicyState(
            num_blocks=self.num_blocks,
            steps=self.steps,
            gen_length=self.gen_length,
            possible_temperatures=self.possible_temperatures, 
            possible_remasking_strategies=self.possible_remasking_strategies
        )

        # this is the policy, so we deep copy the lists because they can be modified
        new.temperature_schedule = self.temperature_schedule.copy()
        new.remasking_strategy_schedule = self.remasking_strategy_schedule.copy()
        new.block_schedule = self.block_schedule.copy()
        new.extra_step_proportions = self.extra_step_proportions.copy()

        return new
    
    def mutate(self, random_select=True, mutation_type=None):
        """
        If random is True, randomly selects a mutation type and applies it. 
        If random is False, uses the provided mutation_type (make sure it is not None and one of the valid types)
        """
        assert mutation_type is None or mutation_type in ['temperature', 'remasking_strategy', 'block_length', 'extra_step_proportion'], "Invalid mutation type."

        new = self.clone()
        
        # Choose mutation type if random is True
        if random_select:
            mutation_type = random.choice(['temperature', 'remasking_strategy', 'block_length', 'extra_step_proportion'])

        index = random.randint(0, self.num_blocks - 1) # randomly select a block to mutate

        # Apply mutation
        if mutation_type == 'temperature':
            options = [t for t in self.possible_temperatures if t != new.temperature_schedule[index]]
            if options:
                new.temperature_schedule[index] = random.choice(options)
        elif mutation_type == 'remasking_strategy':
            options = [s for s in self.possible_remasking_strategies if s != new.remasking_strategy_schedule[index]]
            if options:
                new.remasking_strategy_schedule[index] = random.choice(options)
        elif mutation_type == 'block_length':
            second_index = random.choice([i for i in range(self.num_blocks) if i != index])
            max_length_delta = min(
                new.block_schedule[index] - 1, # ensure there is at least 1 step per block
                new.block_schedule[second_index] - 1
            )
            if max_length_delta > 0:
                delta = random.randint(-max_length_delta, max_length_delta) # select between -max_block_delta and max_block_delta inclusive
                new.block_schedule[index] += delta
                new.block_schedule[second_index] -= delta
        elif mutation_type == 'extra_step_proportion':
            second_index = random.choice([i for i in range(self.num_blocks) if i != index])
            max_proportion_delta = round(min(
                new.extra_step_proportions[index] - 0.01, # ensure there is at least 0.01 proportion per block
                new.extra_step_proportions[second_index] - 0.01
            ), 2) # round to 2 decimal places
            if max_proportion_delta > 0:
                delta = round(random.uniform(-max_proportion_delta, max_proportion_delta), 2) # select between -max_proportion_delta and max_proportion_delta inclusive
                new.extra_step_proportions[index] = round(new.extra_step_proportions[index] + delta, 2)
                new.extra_step_proportions[second_index] = round(new.extra_step_proportions[second_index] - delta, 2)

        return new, mutation_type

if __name__ == '__main__':
    # Example usage
    policy_state = DecodingPolicyState(num_blocks=4, steps=128, gen_length=128)
    policy_state.initialize_default_policy()
    print("Temperature Schedule:", policy_state.temperature_schedule)
    print("Remasking Strategy Schedule:", policy_state.remasking_strategy_schedule)
    print("Block Schedule:", policy_state.block_schedule)
    print("Extra Step Proportions:", policy_state.extra_step_proportions)

    # do 5 mutations
    for i in range(5):
        print(f"Mutated Policy State {i}: ")
        mutated_state, mutation_type = policy_state.mutate(random_select=True, mutation_type=None)
        print(f"Mutation Type ({mutation_type}):")
        print("Temperature Schedule:", mutated_state.temperature_schedule)
        print("Remasking Strategy Schedule:", mutated_state.remasking_strategy_schedule)
        print("Block Schedule:", mutated_state.block_schedule)
        print("Extra Step Proportions:", mutated_state.extra_step_proportions)
        policy_state = mutated_state
        # Update the original policy state to the mutated one for the next iteration