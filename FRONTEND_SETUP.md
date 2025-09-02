# Frontend Setup Instructions

## TypeScript Module Resolution Errors

The TypeScript errors you're seeing:
```
找不到模块"react"或其相应的类型声明。
找不到模块"antd"或其相应的类型声明。
找不到模块"styled-components"或其相应的类型声明。
```

These occur because the node_modules dependencies haven't been installed yet.

## Solution

### 1. Install Node.js
If Node.js is not installed, download and install it from: https://nodejs.org/

### 2. Install Dependencies
Run one of these commands in the project root directory:

```bash
# Using npm
npm install

# Or using yarn (if preferred)
yarn install

# Or run the provided batch file
setup-frontend.bat
```

### 3. Verify Installation
After installation, you should see a `node_modules` folder in the project root.

### 4. Start Development Server
```bash
npm run dev
```

## Dependencies Included
The project includes all necessary dependencies in package.json:
- react ^18.2.0
- antd ^5.10.0  
- styled-components ^6.0.8
- @types/react ^18.2.15
- typescript ^5.0.2

Once dependencies are installed, the TypeScript errors will be resolved.