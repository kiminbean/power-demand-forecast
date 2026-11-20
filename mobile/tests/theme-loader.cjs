const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

// Transpile src/theme/colors.ts so VM-harnessed screens can resolve the theme import.
function loadTheme() {
  const source = fs.readFileSync(path.join(__dirname, '../src/theme/colors.ts'), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
  const themeModule = { exports: {} };
  vm.runInNewContext(code, { module: themeModule, exports: themeModule.exports });
  return themeModule.exports;
}

module.exports = { loadTheme };
