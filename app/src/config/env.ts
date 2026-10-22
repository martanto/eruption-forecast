import { z } from "zod";
export const ENV = z
  .object({ BASE_URL: z.string().startsWith("/") })
  .parse({ BASE_URL: import.meta.env.BASE_URL });
