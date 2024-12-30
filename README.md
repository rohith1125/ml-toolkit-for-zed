# ML Toolkit for Zed

ML Toolkit for Zed is a comprehensive extension for the Zed editor, providing support for machine learning development using TensorFlow and PyTorch.

## Features

- Project creation for both TensorFlow and PyTorch
- Intelligent code completion for ML-specific functions and classes
- Diagnostics for common ML coding issues
- Hover information with integration to official documentation
- Code actions for ML-specific refactoring tasks
- Syntax highlighting for ML-specific code

## Installation

1. Open Zed
2. Go to the Extensions view
3. Search for "ML Toolkit for Zed"
4. Click "Install"

## Usage

### Creating a New ML Project

1. Open the command palette (Cmd+Shift+P on macOS, Ctrl+Shift+P on Windows/Linux)
2. Type "Create ML Project" and select the command
3. Choose the framework (TensorFlow or PyTorch) when prompted
4. A new project structure will be created in your current workspace

### Code Completion

As you type, you'll see ML-specific code completions for both TensorFlow and PyTorch.

### Diagnostics

The extension will provide warnings and errors for common ML coding issues, such as:
- Incorrect framework imports
- Potential memory leaks
- Gradient computation issues

### Hover Information

Hover over ML-specific functions or classes to see documentation from the official TensorFlow or PyTorch websites.

### Code Actions

The extension provides code actions for common ML tasks, such as:
- Converting between TensorFlow and PyTorch
- Optimizing models for inference

To use a code action, click on the lightbulb icon that appears in the editor or use the quick fix keyboard shortcut.

## Development

To contribute to the ML Toolkit for Zed:

1. Clone the repository
2. Install Rust and the Zed extension development tools
3. Run `cargo build` to compile the extension
4. Use the "Run Extension" launch configuration in Zed to test your changes

Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on the development process and coding standards.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

