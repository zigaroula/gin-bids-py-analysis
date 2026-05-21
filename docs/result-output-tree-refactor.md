# Refactor Results Toward Output-Shaped Data

## Goal

The project should stop treating writers as translators between flat processing
results and nested output files. A writer should not know that several separate
fields must be grouped together under a particular HDF5/MATLAB path.

Instead, each processing result should expose data that is already shaped like
the logical output tree. The writer should only serialize that tree.

Target pattern:

```python
tree = result.to_output_tree()
write_hdf5_tree(path, tree)
```

or, when the result stores its output tree directly:

```python
write_hdf5_tree(path, result.output)
```

The same tree should be usable for MATLAB output through the generic MATLAB
serializer.

## Core Principle

The in-memory result model should match the output model as closely as possible.

Avoid result objects like this:

```python
result.condition_a_mean
result.condition_a_sem
result.condition_b_mean
result.condition_b_sem
result.mean_difference
result.difference_sem
result.difference_ci95_low
result.difference_ci95_high
```

Prefer structured objects like this:

```python
result.data.activity.condition_a.mean
result.data.activity.condition_a.sem
result.data.activity.condition_b.mean
result.data.activity.condition_b.sem
result.data.activity.difference.mean
result.data.activity.difference.sem
result.data.activity.difference.ci95_low
result.data.activity.difference.ci95_high
```

or an equivalent mapping:

```python
result.output["data"]["activity"]["difference"]["mean"]
```

The exact container can be a dataclass, Pydantic model, typed mapping, or a
small hybrid, but the ownership rule is the same: the result owns the logical
structure; the writer owns only persistence.

## Responsibilities

### Processing Code

Processing code should construct meaningful structured result components.

Examples of reusable component shapes:

```python
@dataclass
class Estimate:
    mean: np.ndarray
    sem: np.ndarray | None = None
    ci95_low: np.ndarray | None = None
    ci95_high: np.ndarray | None = None


@dataclass
class ConditionPair:
    condition_a: Estimate
    condition_b: Estimate


@dataclass
class StatisticalMap:
    t_values: np.ndarray
    p_values: np.ndarray
    p_values_uncorrected: np.ndarray | None = None
    significant_mask: np.ndarray | None = None
```

These names are examples, not mandatory APIs. The important point is that
related values travel together instead of being scattered as sibling fields.

### Result Objects

Each result type should expose one canonical output representation.

Recommended shape:

```python
@dataclass
class SomeProcessingResult(BaseProcessingResult):
    data: SomeDataTree
    axes: AxesTree
    stats: StatsTree
    meta: MetaTree
    provenance: ProvenanceTree
```

For simple pipelines, these can be dictionaries. For complex pipelines, use
small typed dataclasses with a shared `to_output_tree()` method.

Every result should be able to produce:

```python
{
    "data": ...,
    "axes": ...,
    "stats": ...,
    "meta": ...,
    "provenance": ...,
}
```

Fields that should not be serialized can stay outside that tree, but they
should be clearly marked as runtime-only.

### Writers

Writers should become thin.

Allowed writer responsibilities:

- Validate the result type.
- Choose the serializer based on `output_format`.
- Write companion non-tree files when required, such as TSV audit tables.
- Add format-independent wrapper metadata only if it is truly global.

Avoid writer responsibilities:

- Grouping flat result fields into output paths.
- Renaming result fields into file field names.
- Building pipeline-specific nested structures.
- Duplicating HDF5 and MATLAB structure logic.

Good writer shape:

```python
class SomeProcessingWriter(BaseProcessingWriter):
    def _write_data(self, result, output_path):
        if not isinstance(result, SomeProcessingResult):
            raise TypeError(...)

        tree = result.to_output_tree()
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree)
        else:
            write_hdf5_tree(output_path, tree)
```

If a writer contains many references to domain-specific fields, it is probably
still doing too much.

## Naming Rules

Use stable structural names in the tree.

Prefer:

```text
condition_a
condition_b
difference
mean
sem
ci95_low
ci95_high
```

Avoid using user-facing labels as structural field names:

```text
accepted
rejected
high_confidence
low_confidence
```

User-facing labels should live in metadata:

```python
meta.condition_labels = ["accepted", "rejected"]
```

This prevents writer changes when condition names change.

## Serialization Contract

The canonical output tree must contain only serializable values:

- Scalars: `str`, `bool`, `int`, `float`, numpy scalars.
- Arrays: numpy arrays.
- String arrays: numpy object/string arrays or homogeneous lists of strings.
- Nested mappings or dataclass-converted mappings.
- Optional values omitted or represented consistently.
- Ragged collections represented intentionally, not accidentally.

If a value needs compression, MATLAB naming hints, attributes, or dtype control,
wrap it with serialization metadata near the data definition, not inside a
pipeline-specific writer.

Example:

```python
{
    "stats": {
        "permutations": compressed(permutation_array),
    }
}
```

## Migration Recipe

Apply this process one result type at a time.

1. Identify the current output schema.

   List the HDF5/MATLAB paths currently written and group them into logical
   concepts: data, axes, stats, metadata, provenance, tables, contributions,
   optional heavy arrays.

2. Identify flat field clusters.

   Look for groups of fields that are always written together, such as:

   ```python
   metric_mean
   metric_sem
   metric_ci_low
   metric_ci_high
   ```

   Replace these with a structured component:

   ```python
   metric.mean
   metric.sem
   metric.ci_low
   metric.ci_high
   ```

3. Define the target result shape.

   Decide which parts of the result are canonical output and which are
   runtime-only. Keep the canonical output shaped like the final file tree.

4. Add `to_output_tree()`.

   Initially, this method can convert structured dataclasses to dictionaries.
   The method should be mostly mechanical and should not contain domain-specific
   regrouping of unrelated flat fields.

5. Reduce the writer.

   Replace manual writer logic with:

   ```python
   tree = result.to_output_tree()
   write_hdf5_tree(...)
   write_matlab_tree(...)
   ```

6. Update loaders.

   Load from the new structural paths into the new structured result model.
   Do not restore the old flat shape unless backward-compatible properties are
   explicitly needed.

7. Add compatibility aliases only if needed.

   If existing code still expects old flat fields, provide read-only properties:

   ```python
   @property
   def mean_difference(self):
       return self.data.activity.difference.mean
   ```

   These aliases must be temporary.  Mark them with a comment that names
   every call-site that still depends on them so they are easy to track down.
   Do not use them in new writers, loaders, or analysis code.

   Once all callers have been updated to use the structured path, delete the
   aliases entirely.  A result class is not fully migrated while it still
   carries backward-compat properties.

8. Update tests.

   Tests should assert the new tree shape and the result round-trip, not the old
   flat field names.

9. Remove the compatibility aliases.

   Search for every usage of the aliased names across `src/`, `tests/`, and
   `scripts/`.  Replace each occurrence with the canonical structured path
   (for example `result.contrast.t_values` instead of `result.t_values`, or
   `result.difference.mean` instead of `result.mean_difference`).  Then delete
   the property definitions from the result class.

## Acceptance Criteria

A result type is considered migrated when:

- Its core data is grouped into meaningful nested components.
- Its writer has no domain-specific field regrouping.
- HDF5 and MATLAB are written from the same output tree.
- User labels are metadata, not structural field names.
- Renaming an output object or adding a sibling value requires changing the
  result/tree definition, not the writer.
- Round-trip tests validate both the serialized tree and the loaded result.
- No backward-compat alias properties remain on the result class.

## Anti-Patterns To Remove

Avoid flat aliases on the result after migration is complete:

```python
# BAD — backward-compat alias still referenced in production code
result.mean_difference
result.difference_sem
result.t_values
result.significant_mask
```

Prefer the structured path directly:

```python
result.difference.mean
result.difference.sem
result.contrast.t_values
result.contrast.significant_mask
```

Avoid this:

```python
writer_tree = {
    "difference": {
        "mean": result.mean_difference,
        "sem": result.difference_sem,
    }
}
```

Prefer this:

```python
writer_tree = result.to_output_tree()
```

Avoid this:

```python
fh["means"].create_dataset(result.condition_a, ...)
```

Prefer this:

```python
result.meta.condition_labels = [...]
result.data.activity.condition_a.mean = ...
```

Avoid parallel HDF5/MATLAB builders:

```python
_write_hdf5_specific(...)
_build_matlab_specific(...)
```

Prefer one canonical tree and two generic serializers.

## Practical Default

For each pipeline, start by introducing small dataclasses for logical groups,
then add `to_output_tree()`. Do not try to make one universal result class for
all pipelines immediately.

The useful abstraction is not "one result type for everything"; it is "every
result type exposes an output-shaped tree with the same serialization contract."
