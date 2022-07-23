#! /bin/bash

# gem install copyright-header

copyright-header --add-path ./ppu:./examples \
                 --license-file ./license_template.erb \
                 --copyright-software '' \
                 --copyright-software-description '' \
                 --guess-extension \
                 --copyright-holder 'Ant Group Co., Ltd.' \
                 --copyright-year 2021 \
                 --output-dir ./ \
                 --syntax ./syntax.yml
