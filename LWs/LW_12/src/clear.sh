#!/usr/bin/env bash

for i in $(ls | grep "bigartm.");
do rm -rf "$i";
done