import yaml


def from_config(config):
    return config.pop('class').from_config(config)


def from_yaml(yaml_string):
    config = yaml.load(yaml_string)
    return from_config(config)


class ConfigObject:
    def get_config(self):
        return dict(class_name=self.__class__.__name__)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def to_yaml(self):
        config = self.get_config()
        return yaml.dump(config)

    @classmethod
    def from_yaml(cls, yaml_string):
        config = yaml.load(yaml_string)
        return cls.from_config(config)
