from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="STMSC",
    version="1.0.0",
    description="A novel multi-slice framework for precision 3D spatial domain reconstruction and disease pathology analysis",
    url="https://github.com/bliulab/STMSC",
    author="Daijun Zhang, Ren Qi, Xun Lan, Bin Liu*",
    author_email="djzhang@bliulab.net",
    packages=find_packages(include=["STMSC", "STMSC.*"]),
    install_requires=requirements,  
    keywords=[
        "spatial transcriptomics",
        "three-dimensional spatial reconstruction of tissue",
        "domain identification",
        "cancer"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.10",
    include_package_data=True,
    zip_safe=False,
)

