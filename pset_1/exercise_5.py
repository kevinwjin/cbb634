# Exercise 5: Analysis of MeSH data

# Parse XML
from lxml import etree

print("Parsing XML...")
mesh = etree.parse("/Users/kevin/Downloads/desc2023.xml")
print("Parsing complete.")
root = mesh.getroot()

# Extract values of interest
# Report the DescriptorName associated with DescriptorUI D007154
for record in root.findall(".//*[DescriptorUI='D007154']"):
    name = etree.tostring(record[1], encoding="utf8", method="text").decode().replace("\n", "").strip()
    print(name)

# Report the DescriptorUI associated with DescriptorName "Nervous System Diseases"
for record in root.findall("./*DescriptorName/[String='Nervous System Diseases']/.."):
    print(record[0].text)

# Report DescriptorNames of descendants of both "Nervous System Diseases" and D007154
# Hint: Each item should have tree number C10 and C20
tree_numbers = root.findall(".//TreeNumber")
numbers_of_interest = []
for tree_number in tree_numbers:
    if tree_number.text.startswith("C20") or tree_number.text.startswith("C10"):
        numbers_of_interest.append(tree_number.text)

for number in numbers_of_interest:
    record = root.findall(f".//*[TreeNumber = '{number}']/..")
    name = etree.tostring(record[0][1], encoding="utf8", method="text").decode().replace("\n", "").strip()
    print(name)

# Biomedical explanation: According to the MeSH hierarchy, the above code locates the disease associated with a certain unique identifier,
# the identifier associated with a certain disease, and all conditions that are categorized under both aforementioned diseases. All such
# conditions are classified by the National Library of Medicine as being neuroimmunological disorders, where a body's overly active
# immune system appears to attack its own nervous system, mistaking it for a foreign invader.

# Generic function versions
def get_name(ui):
    for record in root.findall(f".//*[DescriptorUI='{ui}']"):
        name = etree.tostring(record[1], encoding="utf8", method="text").decode().replace("\n", "").strip()
        return name


def get_ui(name):
    for record in root.findall(f"./*DescriptorName/[String='{name}']/.."):
        return record[0].text


def get_descendent_names(tree_number_a, tree_number_b):
    tree_numbers = root.findall(".//TreeNumber")
    numbers_of_interest = []
    for tree_number in tree_numbers:
        if tree_number.text.startswith(tree_number_a) or tree_number.text.startswith(tree_number_b):
            numbers_of_interest.append(tree_number.text)     
    for number in numbers_of_interest:
        record = root.findall(f".//*[TreeNumber = '{number}']/..")
        name = etree.tostring(record[0][1], encoding="utf8", method="text").decode().replace("\n", "").strip()
        print(name)