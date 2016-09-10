from policy import Policy


class MomentumPolicy(Policy):
    def __init__(self, policy, momentum):
        self.policy = policy
        self.momentum = momentum
        self.last_action = 0.0

    def act(self, obs):
        action = self.policy.act(obs)
        action += self.momentum * self.last_action
        self.last_action = action
        return action

    def reset(self):
        self.last_action = 0.0
        return self.policy.reset()

    def _get_config(self):
        config = super(MomentumPolicy, self)._get_config()
        config.update({'policy': self.policy,
                       'momentum': self.momentum})
        return config
