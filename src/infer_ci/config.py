from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)


# Pydantic Model Config
class MyClassConfigPM(BaseModel):
    min_length: int = Field(
        default=2, ge=1, le=1000, description="Minimum length of the list."
    )
    max_length: int = Field(
        default=100, ge=1, le=1000, description="Maximum length of the list."
    )
    min_value: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum value of the each item."
    )
    max_value: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Maximum value of the each item."
    )
    threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Threshold value to filter items."
    )

    model_config = ConfigDict(extra="allow")


ENV_PREFIX = "PMT_"


# Pydantic Cli Config
class MyClassCliConfig(MyClassConfigPM, BaseSettings):

    items: list[float] = Field(..., description="List of float items to be cleaned.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return dotenv_settings, env_settings, file_secret_settings, init_settings

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        cli_parse_args=True,
        cli_enforce_required=True,
    )


__all__ = [
    "ENV_PREFIX",
    "MyClassConfigPM",
    "MyClassCliConfig",
]
