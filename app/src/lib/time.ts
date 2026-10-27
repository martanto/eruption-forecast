export const DAY_SECONDS = 86_400;
export function epochSeconds(date: string): number {
  return Date.parse(`${date}T00:00:00Z`) / 1000;
}
export function formatDate(date: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(epochSeconds(date) * 1000);
}
// Accepts either a YYYY-MM-DD string or an epoch in milliseconds.
export function shortDate(date: string | number): string {
  const time = typeof date === "string" ? epochSeconds(date) * 1000 : date;
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
  }).format(time);
}
export function utcStamp(time: number): string {
  return new Date(time).toISOString().slice(0, 16).replace("T", " ");
}
export function dateOffset(date: string, origin: string): number {
  return (epochSeconds(date) - epochSeconds(origin)) / DAY_SECONDS;
}
