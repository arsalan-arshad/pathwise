#!/usr/bin/env node
import { main } from "./cli-core.js";

process.exitCode = await main(process.argv.slice(2), (s) => console.log(s), (s) => console.error(s));
