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
    
    def mutate(self):
        new = self.clone()
        pass

if __name__ == '__main__':
    # Example usage
    policy_state = DecodingPolicyState(num_blocks=4, steps=128, gen_length=128)
    policy_state.initialize_default_policy()
    print("Temperature Schedule:", policy_state.temperature_schedule)
    print("Remasking Strategy Schedule:", policy_state.remasking_strategy_schedule)
    print("Block Schedule:", policy_state.block_schedule)
    print("Extra Step Proportions:", policy_state.extra_step_proportions)

    cloned_state = policy_state.clone()
    cloned_state.temperature_schedule[0] = 0.5  # Modify cloned state
    # Output should show that the original state remains unchanged and that the cloned state reflects the modification
    print("Original Temperature Schedule after cloning:", policy_state.temperature_schedule)
    print("Cloned Temperature Schedule after modification:", cloned_state.temperature_schedule)
    print("Original Remasking Strategy Schedule after cloning:", policy_state.remasking_strategy_schedule)
    print("Cloned Remasking Strategy Schedule after modification:", cloned_state.remasking_strategy_schedule)
    print("Original Block Schedule after cloning:", policy_state.block_schedule)
    print("Cloned Block Schedule after modification:", cloned_state.block_schedule)
    print("Original Extra Step Proportions after cloning:", policy_state.extra_step_proportions)
    print("Cloned Extra Step Proportions after modification:", cloned_state.extra_step_proportions)