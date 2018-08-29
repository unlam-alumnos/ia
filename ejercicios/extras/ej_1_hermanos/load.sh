#!/bin/bash
# keyspace is the graph name to load data

keyspace='ej_1_hermanos'

graql console -f ./ontologies/base_ontology.gql -k $keyspace 
graql console -f ./rules/base_rules.gql -k $keyspace
graql console -f ./data/data.gql -k $keyspace