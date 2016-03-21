import yaml


def from_config(config):
    config = dict(config)  # shallow copy
    return config.pop('class').from_config(config)


def from_yaml(yaml_string):
    config = yaml.load(yaml_string)
    return from_config(config)


class ConfigObject:
    def get_config(self):
        return dict(class_name=self.__class__.__name__)

    @classmethod
    def from_config(cls, config):
        # TODO pop class?
        return cls(**config)

    def to_yaml(self, *args, **kwargs):
        config = self.get_config()
        return yaml.dump(config, *args, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_string):
        config = yaml.load(yaml_string)
        return cls.from_config(config)
