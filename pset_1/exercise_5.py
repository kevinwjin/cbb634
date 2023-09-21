# Exercise 5: Analysis of MeSH data

# Download and parse XML
import xml.etree.ElementTree as ET

print("Parsing XML...")
mesh = ET.parse("/Users/kevin/Downloads/desc2023.xml")
print("Parsing complete.")

root = mesh.getroot()
print(root.tag, root.attrib)

# Extract values of interest

# Biomedical explanation:

# Generic function versions
def get_name(ui):
    pass


def get_ui(name):
    pass


def get_descendent_names(ui, name):
    pass


# Test generic function versions
