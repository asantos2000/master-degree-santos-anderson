from graphviz import Digraph

# Create the diagram
dot = Digraph(comment="CNL to SBVR PIM Mapping Example")

# Nodes
dot.node("CNL", "CNL Sentence:\n\"Each customer must have at least one active account.\"")
dot.node("PARSE", "Parsing & Vocabulary Binding:\n- customer → noun concept\n- account → noun concept\n- have → fact type\n- must → obligation\n- at least one → quantifier\n- active → property")
dot.node("MAP", "Mapping to SBVR Semantic Formulation (PIM):\n- Concepts: customer, account\n- Fact Type: customer has account\n- Quantification: at least one active account per customer\n- Modality: obligation")
dot.node("FORM", "Formal Logical Representation (PIM):\n∀c (Customer(c) → O∃a (Account(a) ∧ Active(a) ∧ Has(c,a)))")
dot.node("PSM", "Possible PSM Transformations:\n- SQL constraint\n- BRMS rule\n- Application validation code")

# Edges
dot.edge("CNL", "PARSE")
dot.edge("PARSE", "MAP")
dot.edge("MAP", "FORM")
dot.edge("FORM", "PSM")

# Render the diagram to a file
output_path = 'cnl_sbvr_pim_example'
dot.render(output_path, format='png', cleanup=True)

output_path + ".png"
