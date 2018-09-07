#!/bin/bash
# keyspace is the graph name to load data

keyspace='ej_1_hermanos_solucion_profesor'

graql console -f family_ontology.gql -k $keyspace 
graql console -f family_rules.gql -k $keyspace
graql console -f family_data.gql -k $keyspace