import { useEffect, useState } from "react";
import { RotateCcw, Search } from "lucide-react";
import { useWorkspace, type Schema } from "../../hooks/useWorkspace";
import { label, updateAt, type JsonObject } from "../../lib/configuration";

interface Props {
  definition: string;
  value: JsonObject;
  defaults: JsonObject;
  onChange: (value: JsonObject) => void;
  overrides?: boolean;
  search?: boolean;
  include?: string[];
  fieldOptions?: Record<
    string,
    {
      placeholder?: string;
      hint?: string;
      control?: (
        id: string,
        value: unknown,
        change: (value: unknown) => void,
      ) => React.ReactNode;
    }
  >;
  showProvenance?: boolean;
}

function effectiveSchema(
  schema: Schema,
  definitions: Record<string, Schema>,
  value: unknown,
): Schema {
  if (schema.$ref) return definitions[schema.$ref.split("/").at(-1)!];
  if (schema.anyOf) {
    const candidates = schema.anyOf.filter((item) => item.type !== "null");
    return (
      candidates.find(
        (item) =>
          item.type ===
          (Array.isArray(value)
            ? "array"
            : typeof value === "number"
              ? "number"
              : typeof value === "boolean"
                ? "boolean"
                : "string"),
      ) ?? candidates[0]
    );
  }
  return schema;
}

function JsonInput({
  value,
  onChange,
  id,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
  id: string;
}) {
  const serialized = value == null ? "" : JSON.stringify(value, null, 2);
  const [draft, setDraft] = useState(serialized);
  const [error, setError] = useState("");
  useEffect(() => {
    setDraft(serialized);
    setError("");
  }, [serialized]);
  return (
    <>
      <textarea
        id={id}
        value={draft}
        rows={3}
        spellCheck={false}
        onChange={(event) => {
          const text = event.target.value;
          setDraft(text);
          try {
            if (text.trim()) JSON.parse(text);
            setError("");
          } catch {
            setError('Enter valid JSON, for example ["rsam", "dsar"].');
          }
        }}
        onBlur={() => {
          try {
            onChange(draft.trim() ? JSON.parse(draft) : null);
            setError("");
          } catch {
            setError('Enter valid JSON, for example ["rsam", "dsar"].');
          }
        }}
        aria-invalid={!!error}
        aria-describedby={error ? `${id}-error` : undefined}
      />
      {error && (
        <span id={`${id}-error`} className="field-error">
          {error}
        </span>
      )}
    </>
  );
}

export function ParameterForm({
  definition,
  value,
  defaults,
  onChange,
  overrides = false,
  search = true,
  include,
  fieldOptions = {},
  showProvenance = true,
}: Props) {
  const { catalog, artifacts } = useWorkspace();
  const [query, setQuery] = useState("");
  const definitions = catalog.schema.$defs;
  const schema = definitions[definition];
  if (!schema)
    return <p role="alert">Parameter contract unavailable: {definition}</p>;
  const fields = (
    node: Schema,
    current: JsonObject,
    inherited: JsonObject,
    path: string[] = [],
  ): React.ReactNode =>
    Object.entries(node.properties ?? {}).map(([name, raw]) => {
      if (path.length === 0 && include && !include.includes(name)) return null;
      const options = fieldOptions[name];
      const next = [...path, name];
      const supplied = Object.prototype.hasOwnProperty.call(current, name);
      const display = supplied
        ? current[name]
        : (inherited[name] ?? raw.default);
      const shape = effectiveSchema(raw, definitions, display);
      const id = `${definition}-${next.join("-")}`;
      const change = (content: unknown) =>
        onChange(updateAt(value, next, content));
      if (shape.properties) {
        return (
          <details
            className="parameter-group"
            key={id}
            open={query || name === "constructor" ? true : undefined}
          >
            <summary>
              {label(name)}
              <span>{Object.keys(shape.properties).length} parameters</span>
            </summary>
            <div className="form-grid">
              {fields(
                shape,
                (current[name] as JsonObject) ?? {},
                (inherited[name] as JsonObject) ?? {},
                next,
              )}
            </div>
          </details>
        );
      }
      if (
        query &&
        !`${name} ${label(name)} ${raw.description ?? ""}`
          .toLowerCase()
          .includes(query.toLowerCase())
      )
        return null;
      const nullable = raw.anyOf?.some((option) => option.type === "null");
      const enums =
        shape.enum ?? raw.anyOf?.find((option) => option.enum)?.enum;
      const pathInput =
        /(_dir|_path|_csv|tremor_data|^model$|select_features)$/.test(name);
      return (
        <div
          className={`field ${shape.type === "array" ? "wide" : ""}`}
          key={id}
        >
          <div className="field-label">
            <label htmlFor={id}>{label(name)}</label>
            {supplied && (
              <button
                className="reset-field"
                type="button"
                aria-label={`Reset ${label(name)}`}
                title="Use inherited value"
                onClick={() => onChange(updateAt(value, next, undefined, true))}
              >
                <RotateCcw size={12} />
              </button>
            )}
          </div>
          {options?.control ? (
            options.control(id, display, change)
          ) : shape.type === "boolean" ? (
            <label className="checkbox-row">
              <input
                id={id}
                type="checkbox"
                checked={display === true}
                onChange={(event) => change(event.target.checked)}
              />
              <span>{display === true ? "Enabled" : "Disabled"}</span>
            </label>
          ) : enums ? (
            <select
              id={id}
              value={display == null ? "__null" : String(display)}
              onChange={(event) =>
                change(
                  event.target.value === "__null" ? null : event.target.value,
                )
              }
            >
              {nullable && <option value="__null">None / inherit</option>}
              {enums.map((option) => (
                <option key={String(option)} value={String(option)}>
                  {String(option)}
                </option>
              ))}
            </select>
          ) : shape.type === "array" ? (
            <JsonInput key={id} id={id} value={display} onChange={change} />
          ) : (
            <input
              id={id}
              list={pathInput ? `${id}-artifacts` : undefined}
              type={
                shape.type === "integer" || shape.type === "number"
                  ? "number"
                  : name.endsWith("date")
                    ? "date"
                    : "text"
              }
              step={shape.type === "number" ? "any" : undefined}
              value={display == null ? "" : String(display)}
              placeholder={
                options?.placeholder ?? (nullable ? "None / inherit" : "")
              }
              aria-describedby={options?.hint ? `${id}-hint` : undefined}
              onChange={(event) =>
                change(
                  event.target.value === "" && nullable
                    ? null
                    : ["integer", "number"].includes(shape.type ?? "")
                      ? Number(event.target.value)
                      : event.target.value,
                )
              }
            />
          )}
          {pathInput && (
            <datalist id={`${id}-artifacts`}>
              {artifacts
                .filter((item) => /\.(csv|parquet|pkl|json)$/.test(item.path))
                .map((item) => (
                  <option key={item.id} value={item.path} />
                ))}
            </datalist>
          )}
          {raw.anyOf?.some((option) => option.type === "array") &&
            raw.anyOf?.some((option) => option.type === "string") && (
              <button
                type="button"
                className="text-button null-control"
                onClick={() => change(Array.isArray(display) ? "" : [])}
              >
                {Array.isArray(display)
                  ? "Use a path or single value"
                  : "Use a list of values"}
              </button>
            )}
          {raw.anyOf?.some((option) => option.type === "number") &&
            raw.anyOf?.some((option) => option.type === "string") && (
              <button
                type="button"
                className="text-button null-control"
                onClick={() =>
                  change(typeof display === "number" ? "auto" : 0.75)
                }
              >
                {typeof display === "number"
                  ? "Use a named strategy"
                  : "Use a numeric ratio"}
              </button>
            )}
          {(options?.hint || showProvenance) && (
            <span id={`${id}-hint`} className="field-hint">
              {options?.hint ??
                (overrides
                  ? supplied
                    ? "Scenario override"
                    : "Inherited from shared settings"
                  : next.length
                    ? `${next.join(".")} · ${supplied ? "configured" : "default"}`
                    : (raw.description ?? name))}
            </span>
          )}
          {showProvenance && nullable && display != null && (
            <button
              type="button"
              className="text-button null-control"
              onClick={() => change(null)}
            >
              Set to none
            </button>
          )}
        </div>
      );
    });
  return (
    <div className="parameter-form">
      {search && (
        <label className="search-field">
          <Search size={16} />
          <input
            aria-label="Find a parameter"
            placeholder="Find a parameter…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
      )}
      <div className="form-grid">{fields(schema, value, defaults)}</div>
    </div>
  );
}
