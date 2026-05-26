# Contributing to Brain Tumor Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Error messages or logs

### Suggesting Enhancements

For feature requests:
- Describe the feature clearly
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test your changes**:
   ```bash
   python validate_dataset.py
   python train.py --epochs 5  # Quick test
   ```
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add: feature description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and small

## ✅ Testing

Before submitting:
- Test on sample data
- Verify all scripts run without errors
- Check that documentation is updated
- Ensure code is properly formatted

## 📚 Documentation

When adding features:
- Update README.md if needed
- Add docstrings to new functions
- Update DATASET_SETUP.md for data-related changes
- Include usage examples

## 🎯 Areas for Contribution

- **Model improvements**: New architectures, hyperparameter tuning
- **Data augmentation**: Additional augmentation techniques
- **Visualization**: Better plots and visualizations
- **Performance**: Speed optimizations
- **Documentation**: Tutorials, examples, guides
- **Testing**: Unit tests, integration tests
- **Bug fixes**: Fix reported issues

## 💡 Development Setup

1. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Set up pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🔍 Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

## 📧 Questions?

If you have questions:
- Open a discussion on GitHub
- Check existing issues
- Contact maintainers

Thank you for contributing! 🎉