#!/bin/bash
# keyspace is the graph name to load data

keyspace=$1

graql console -f ./ontologies/base_ontology.gql -k $keyspace 
graql console -f ./rules/base_rules.gql -k $keyspace
graql console -f ./data/elect_device_data.gql -k $keyspace