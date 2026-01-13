from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError

JsonSchema = dict[str, Any]


@dataclass(init=False)
class JsonSchemaTransformer(ABC):
    """Walks a JSON schema, applying transformations to it at each level.

    Note: We may eventually want to rework tools to build the JSON schema from the type directly, using a subclass of
    pydantic.json_schema.GenerateJsonSchema, rather than making use of this machinery.
    """
    def __init__(
        self,
        schema: JsonSchema,
        *,
        strict: bool | None = None,
        prefer_inlined_defs: bool = False,
        simplify_nullable_unions: bool = False,
    ):
        self.schema = schema
        self.strict = strict
        self.is_strict_compatible = True  # Can be set to False by subclasses to set `strict` on `ToolDefinition` when set not set by user explicitly

        self.prefer_inlined_defs = prefer_inlined_defs
        self.simplify_nullable_unions = simplify_nullable_unions

        # Use views and avoid copying dicts/allocations unnecessarily
        self.defs: dict[str, JsonSchema] = self.schema.get('$defs', {})
        self.refs_stack: list[str] = []
        self.recursive_refs = set[str]()

    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Make changes to the schema."""
        return schema

    def walk(self) -> JsonSchema:
        # Optimization: Avoid deepcopy unless $defs exists or we are going to mutate
        # We only need to copy the 'outer' schema except $defs (possibly handled separately)
        # Instead, make a shallow copy and update/finalize after handling
        orig_schema = self.schema
        # Only copy non-defs keys for initial schema; $defs will be handled separately if needed
        if '$defs' in orig_schema:
            schema = {k: v for k, v in orig_schema.items() if k != '$defs'}
        else:
            schema = dict(orig_schema)

        handled = self._handle(schema)

        # Fast path: prefer_inlined_defs = False and self.defs exists
        if not self.prefer_inlined_defs and self.defs:
            # Only handle $defs after the main schema
            # In almost all cases, self.defs is a dict[str, JsonSchema]. Use dict comprehension directly.
            # Avoid creating intermediate dicts; use generator if possible, but here dict-comprehension is straightforward.
            handled['$defs'] = {
                k: self._handle(v) for k, v in self.defs.items()
            }

        elif self.recursive_refs:  # pragma: no cover
            # If preferring inlined defs but recursion detected, keep $defs+$ref
            defs = {key: self.defs[key] for key in self.recursive_refs}
            root_ref = orig_schema.get('$ref')
            root_key = None if root_ref is None else _fast_defs_ref_strip(root_ref)
            if root_key is None:
                root_key = orig_schema.get('title', 'root')
                while root_key in defs:
                    root_key = f'{root_key}_root'
            defs[root_key] = handled
            return {'$defs': defs, '$ref': f'#/$defs/{root_key}'}

        return handled

    def _handle(self, schema: JsonSchema) -> JsonSchema:
        nested_refs = 0

        # Fast str.replace() is much faster than re.sub here for a fixed pattern
        if self.prefer_inlined_defs:
            while True:
                ref = schema.get('$ref')
                if not ref:
                    break
                key = _fast_defs_ref_strip(ref)
                if key in self.refs_stack:
                    self.recursive_refs.add(key)
                    break  # recursive ref can't be unpacked
                self.refs_stack.append(key)
                nested_refs += 1

                def_schema = self.defs.get(key)
                if def_schema is None:  # pragma: no cover
                    raise UserError(f'Could not find $ref definition for {key}')
                schema = def_schema

        type_ = schema.get('type')
        if type_ == 'object':
            schema = self._handle_object(schema)
        elif type_ == 'array':
            schema = self._handle_array(schema)
        elif type_ is None:
            schema = self._handle_union(schema, 'anyOf')
            schema = self._handle_union(schema, 'oneOf')

        schema = self.transform(schema)

        if nested_refs > 0:
            del self.refs_stack[-nested_refs:]

        return schema

    def _handle_object(self, schema: JsonSchema) -> JsonSchema:
        if properties := schema.get('properties'):
            handled_properties = {}
            for key, value in properties.items():
                handled_properties[key] = self._handle(value)
            schema['properties'] = handled_properties

        if (additional_properties := schema.get('additionalProperties')) is not None:
            if isinstance(additional_properties, bool):
                schema['additionalProperties'] = additional_properties
            else:
                schema['additionalProperties'] = self._handle(additional_properties)

        if (pattern_properties := schema.get('patternProperties')) is not None:
            handled_pattern_properties = {}
            for key, value in pattern_properties.items():
                handled_pattern_properties[key] = self._handle(value)
            schema['patternProperties'] = handled_pattern_properties

        return schema

    def _handle_array(self, schema: JsonSchema) -> JsonSchema:
        if prefix_items := schema.get('prefixItems'):
            schema['prefixItems'] = [self._handle(item) for item in prefix_items]

        if items := schema.get('items'):
            schema['items'] = self._handle(items)

        return schema

    def _handle_union(self, schema: JsonSchema, union_kind: Literal['anyOf', 'oneOf']) -> JsonSchema:
        members = schema.get(union_kind)
        if not members:
            return schema

        handled = [self._handle(member) for member in members]

        # convert nullable unions to nullable types
        if self.simplify_nullable_unions:
            handled = self._simplify_nullable_union(handled)

        if len(handled) == 1:
            # In this case, no need to retain the union
            return handled[0]

        # If we have keys besides the union kind (such as title or discriminator), keep them without modifications
        schema = schema.copy()
        schema[union_kind] = handled
        return schema

    @staticmethod
    def _simplify_nullable_union(cases: list[JsonSchema]) -> list[JsonSchema]:
        # TODO: Should we move this to relevant subclasses? Or is it worth keeping here to make reuse easier?
        if len(cases) == 2 and {'type': 'null'} in cases:
            # Find the non-null schema
            non_null_schema = next(
                (item for item in cases if item != {'type': 'null'}),
                None,
            )
            if non_null_schema:
                # Create a new schema based on the non-null part, mark as nullable
                new_schema = deepcopy(non_null_schema)
                new_schema['nullable'] = True
                return [new_schema]
            else:  # pragma: no cover
                # they are both null, so just return one of them
                return [cases[0]]

        return cases


class InlineDefsJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema to inline $defs."""

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


def _fast_defs_ref_strip(ref: str) -> str:
    """Replace the regex usage for stripping '#/$defs/' prefix."""
    # Assumes only used for prefix '#/$defs/', which benchmarks far faster as a string slice or str.replace
    prefix = "#/$defs/"
    if ref.startswith(prefix):
        return ref[len(prefix):]
    return ref
