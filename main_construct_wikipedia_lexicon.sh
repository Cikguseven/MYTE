#! /bin/bash

shopt -s nullglob


LANGUAGES=('af' 'am' 'az' 'be' 'bg' 'bn' 'ca' 'ceb' 'co' 'cs' 'cy' 'da' 'de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'sr' 'st' 'su' 'sv' 'sw' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu')
# LANGUAGES=('ar' 'bo' 'et'  'fa' 'haw' 'he' 'hi' 'ja' 'ko' 'lo' 'lt' 'mi'  'ny' 'sm' 'st' 'ta' 'xh')
# LANGUAGES=('es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ar' 'ja' 'ko' 'te' 'ta' 'bo' 'si')
LANGUAGE_COUNT=$(( ${#LANGUAGES[@]} - 1 ))


sbatch --array=0-${LANGUAGE_COUNT} job_construct_wikipedia_lexicon.sh
