#!/bin/bash
# keyspace is the graph name to load data

keyspace='ej_2_ontologias'

# Cargando ontología, reglas y datos del ejercicio anterior
graql console -f ../ej_1/ontologies/base_ontology.gql -k $keyspace 
graql console -f ./ontologies/base_ontology.gql -k $keyspace 

graql console -f ../ej_1/rules/base_rules.gql -k $keyspace
graql console -f ./rules/base_rules.gql -k $keyspace

graql console -f ../ej_1/data/data.gql -k $keyspace
graql console -f ./data/data.gql -k $keyspace
