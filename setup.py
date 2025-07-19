from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="halcyon-journal",
    version="1.0.0",
    author="Halcyon Journal",
    description="A privacy-focused, local-only interactive journaling app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/halcyon-journal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "halcyon=journal_app:main",
        ],
    },
    keywords="journal, privacy, local, ai, ollama, markdown",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/halcyon-journal/issues",
        "Source": "https://github.com/yourusername/halcyon-journal",
    },
) 