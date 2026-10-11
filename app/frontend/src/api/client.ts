import type { components } from "./generated/schema";

export type Configuration = components["schemas"]["Configuration"];
export type Job = components["schemas"]["Job"];
export type JobKind = components["schemas"]["JobRequest"]["kind"];
export type Setup = components["schemas"]["Setup"];
export interface Artifact {
  id: string;
  path: string;
  job_id: string | null;
  created_at: string;
}
export interface TableData {
  columns: string[];
  rows: Record<string, string | number | null>[];
  total: number | null;
  sampled: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    method: method ?? (body === undefined ? "GET" : "POST"),
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `${response.status}: ${response.statusText}`;
    try {
      const error = await response.json();
      message =
        typeof error.detail === "string"
          ? error.detail
          : error.detail
              .map(
                (item: { loc?: string[]; msg: string }) =>
                  `${item.loc?.join(".") ?? ""}: ${item.msg}`,
              )
              .join("\n");
    } catch {
      /* Preserve HTTP context when a proxy returns non-JSON. */
    }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}
