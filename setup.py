from setuptools import setup, find_packages

setup(
    name="kd-ar-stream",
    version="1.0.3",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "kd_ar_stream": ["data/*.txt"]
    },
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0"
    ],
    python_requires=">=3.10",
    description="KD-AR Stream: Real-Time Data Stream Clustering with Adaptive Radius",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/senolali/kd-ar-stream",
    author="Ali Şenol",
    author_email="alisenol@tarsus.edu.tr",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Homepage": "https://github.com/senolali/kd-ar-stream",
        "Documentation": "https://github.com/senolali/kd-ar-stream",
        "Source": "https://github.com/senolali/kd-ar-stream",
        "Issues": "https://github.com/senolali/kd-ar-stream",
        "Paper": "https://doi.org/10.17341/gazimmfd.467226"
    }
)  


