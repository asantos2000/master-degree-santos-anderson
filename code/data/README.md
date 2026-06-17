# Data files

- cfr2sbvr_db - CFR2SBVR DB used for cfr2sbvr-inspect application
  - db_objects - Views used by the application
  - metadata - Metadata used by the application
  - database_v4.db - Database used by the application loaded with data from run_v4
  - database_v5.db - Database used by the application loaded with data from run_v5
- checkpoints - Runtime directory for notebook checkpoints
- checkpoints_classification - Checkpoints after run the classification notebook
- checkpoints_evaluation - Checkpoints after run the evaluation notebook
- checkpoints_extraction - Checkpoints after run the extraction notebook
- checkpoints_transform - Checkpoints after run the transform notebook
- run_v4 - Checkpoints version 4
- run_v5 - Checkpoints version 5
- temp - Temporary files
- classify_subtypes-v2.ttl - Used for KG
- classify_subtypes.json - Used for classification and transform notebook
- classify_subtypes.ttl - Used for KG
- classify_subtypes.yaml - Used for classification and transform notebook
- Code_Federal_Regulations.ttl - Used for KG
- documents_true_table.json - The true table used for all notebooks
- FRO_CFR_Title_17_Part_275.ttl - Used for KG
- MultipleVocabularyFacility.rdf - Used for KG
- prod-fibo-quickstart-2024Q2.ttl - Used for KG
- prod-fibo-quickstart-2024Q3.ttl - Used for KG
- README.md
- sbvr-dtc-19-05-32-ontology-v1-title.ttl - Used for KG
- sbvr-dtc-19-05-32-ontology-v1.ttl - Used for KG
- sbvr-dtc-19-05-32.xsd - Used for KG
- sbvr-dtc-19-05-33.xml - Used for KG
- US_LegalReference.ttl - Used for KG
- witt_examples.yaml - Used for classification and transform notebook
- witt_rules_taxonomy_v1.ttl - Used for KG
- witt_subtemplates.yaml - Used for classification and transform notebook
- witt_template_subtemplate_relationship.yaml - Used for classification and transform notebook
- witt_templates.yaml - Used for classification and transform notebook

## Database versions

### Database v4
Database used by the application loaded with data from run_v4 and views created from code/data/cfr2sbvr_db/db_objects

The relationship of classify elements, and transform elements to extract elements is defined is by true table checkpoint.

### Database v5
Database used by the application loaded with data from run_v5 and views created from code/data/cfr2sbvr_db/db_objects_v5

The relationship of classify elements, and transform elements to extract elements is defined is by the checkpoint.