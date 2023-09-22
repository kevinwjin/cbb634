# Exercise 5: Analysis of MeSH data

# Parse XML
import xml.etree.ElementTree as ET

print("Parsing XML...")
mesh = ET.parse("/Users/kevin/Downloads/desc2023.xml")
print("Parsing complete.")
root = mesh.getroot()

# Extract values of interest
# Report the DescriptorName associated with DescriptorUI D007154
for record in root.findall(".//*[DescriptorUI='D007154']"):
    name = ET.tostring(record[1], encoding="utf8", method="text").decode().replace("\n", "").strip()
    print(name)

# Report the DescriptorUI associated with DescriptorName "Nervous System Diseases"
for record in root.findall(".//*[DescriptorName='Nervous System Diseases']"):
    ui = ET.tostring(record[0], encoding="utf8", method="text").decode().replace("\n", "").strip()
    print(ui)

# for child in root.iter():
#     if child[0].text == "D007154":
#         print(child[1].text)

# items = list(mesh.iter())
# for i, item in enumerate(items):
#     if item.text == "D007154":
#         name == items[i + 1]

# for sec in parent.iter("sec"):
#     for title in sec.iter("title"):
#         text = title.text
#         if text and "methods" in text:
#                 print("**title: " + text + " **** sec id : " + sec.get("id, "))

# Biomedical explanation:


# Generic function versions
def get_name(ui):
    pass


def get_ui(name):
    pass


def get_descendent_names(ui, name):
    pass


# Test generic function versions
