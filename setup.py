from setuptools import setup, find_packages

long_description = """
A Convex-Concave Procedure-based SQP method from Signal Temporal Logic (STL) specifications
"""

setup(name='STLCCP',
      version='0.1.0',
      description='A Convex-Concave Procedure-based SQP method from Signal Temporal Logic (STL) specifications',
      long_description=long_description,
      long_description_content_type='text/markdown',
      project_urls ={
          "Source Code": "https://github.com/",
},
      author='Yoshinari Takayama',
      author_email='yoshijbbsk1121@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'treelib'],
      python_requires='>=3.8',
      zip_safe=False)
