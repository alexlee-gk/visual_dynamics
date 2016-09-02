from policy import Policy


class TargetPolicy(Policy):
    def get_target_state(self):
        raise NotImplementedError
