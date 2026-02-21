{
  "extends": "eslint:recommended",
  "ignorePatterns": ["dist", ".eslintrc.cjs"],
  "parser": "@typescript-eslint/parser",
  "plugins": ["react-refresh"],
  "rules": {
    "react-refresh/only-exports-components": [
      "warn",
      { "allowConstantExport": true }
    ]
  }
}
