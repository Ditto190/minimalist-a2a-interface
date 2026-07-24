"""
Flow state management — unstructured (dict) and structured (Pydantic) state.

Every flow carries a state object that survives across steps and can be
persisted/restored. Unstructured flows use a plain ``dict`` with an
auto-generated ``id`` key; structured flows use a Pydantic model, to which an
``id`` field is added automatically when the user model does not declare one.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field, create_model

from mangaba.core.exceptions import FlowError

log = logging.getLogger(__name__)

#: Type alias for anything usable as flow state.
StateLike = Union[Dict[str, Any], BaseModel]

_ID_MODEL_CACHE: Dict[int, Type[BaseModel]] = {}


def new_state_id() -> str:
    """Return a fresh UUID4 hex string used as a state/flow identifier."""
    return uuid.uuid4().hex


class FlowState(BaseModel):
    """Convenience base for structured flow state.

    Subclassing is optional — any Pydantic model works as a state model — but
    inheriting from :class:`FlowState` guarantees the ``id`` field is present
    and typed.

    Example::

        class ReportState(FlowState):
            topic: str = ""
            sections: list[str] = []

        class ReportFlow(Flow[ReportState]):
            ...
    """

    id: str = Field(default_factory=new_state_id)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def ensure_id_field(model: Type[BaseModel]) -> Type[BaseModel]:
    """Return a model equivalent to ``model`` but guaranteed to have an ``id``.

    If the supplied model already declares an ``id`` field it is returned
    unchanged. Otherwise a subclass adding ``id: str`` (auto-generated) is
    created once and cached.

    Args:
        model: Any Pydantic v2 model class.

    Returns:
        A model class exposing an ``id`` field.

    Example::

        Patched = ensure_id_field(MyState)
        state = Patched(topic="AI")
        assert state.id
    """
    if "id" in model.model_fields:
        return model

    cached = _ID_MODEL_CACHE.get(id(model))
    if cached is not None:
        return cached

    patched = create_model(  # type: ignore[call-overload]
        f"{model.__name__}WithId",
        __base__=model,
        id=(str, Field(default_factory=new_state_id)),
    )
    _ID_MODEL_CACHE[id(model)] = patched
    log.debug("Added auto 'id' field to state model %s", model.__name__)
    return patched


def create_state(
    state_model: Optional[Type[BaseModel]] = None,
    initial: Optional[Dict[str, Any]] = None,
) -> StateLike:
    """Build a fresh state object.

    Args:
        state_model: Optional Pydantic model. When ``None`` an unstructured
            dict state is produced.
        initial: Initial values merged into the new state.

    Returns:
        A dict (unstructured) or a model instance (structured), always with an
        ``id``.

    Raises:
        FlowError: If ``initial`` is not valid for ``state_model``.

    Example::

        state = create_state(None, {"topic": "AI"})
        assert "id" in state
    """
    values = dict(initial or {})

    if state_model is None:
        values.setdefault("id", new_state_id())
        return values

    model = ensure_id_field(state_model)
    try:
        return model(**values)
    except Exception as exc:  # pydantic ValidationError and friends
        raise FlowError(
            f"Could not initialise state model '{state_model.__name__}': {exc}",
            cause=exc,
        ) from exc


def get_state_id(state: StateLike) -> str:
    """Return the ``id`` carried by a state object (empty string if absent)."""
    if isinstance(state, BaseModel):
        return str(getattr(state, "id", "") or "")
    return str(state.get("id", "") or "")


def set_state_id(state: StateLike, value: str) -> None:
    """Force the ``id`` of a state object (used when forking/resuming)."""
    if isinstance(state, BaseModel):
        try:
            setattr(state, "id", value)
        except Exception:  # frozen models
            object.__setattr__(state, "id", value)
    else:
        state["id"] = value


def state_to_dict(state: StateLike) -> Dict[str, Any]:
    """Serialise a state object to a plain, JSON-friendly dict."""
    if isinstance(state, BaseModel):
        return state.model_dump(mode="json")
    return dict(state)


def state_from_dict(
    data: Dict[str, Any],
    state_model: Optional[Type[BaseModel]] = None,
) -> StateLike:
    """Rebuild a state object from a dict produced by :func:`state_to_dict`.

    Args:
        data: Previously serialised state.
        state_model: Model to rehydrate into, or ``None`` for dict state.

    Returns:
        The restored state object.

    Raises:
        FlowError: If the stored payload no longer matches ``state_model``.
    """
    if state_model is None:
        restored = dict(data)
        restored.setdefault("id", new_state_id())
        return restored

    model = ensure_id_field(state_model)
    try:
        return model(**data)
    except Exception as exc:
        raise FlowError(
            f"Persisted state is incompatible with '{state_model.__name__}': {exc}",
            cause=exc,
        ) from exc


def update_state(state: StateLike, values: Dict[str, Any]) -> StateLike:
    """Merge ``values`` into ``state`` in place and return it.

    Unknown keys are written to dict state, and skipped (with a debug log) for
    structured state that does not allow extra fields.

    Example::

        update_state(self.state, {"topic": "AI"})
    """
    if not values:
        return state

    if isinstance(state, BaseModel):
        allow_extra = state.model_config.get("extra") == "allow"
        for key, value in values.items():
            if key in type(state).model_fields or allow_extra:
                setattr(state, key, value)
            else:
                log.debug("Ignoring input '%s' — not a field of %s", key, type(state).__name__)
        return state

    state.update(values)
    return state
