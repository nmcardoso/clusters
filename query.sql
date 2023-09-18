SELECT dual_g.ID, dual_g.RA, dual_g.DEC, dual_g.Field, dual_g.g_auto, dual_r.r_auto, dual_i.i_auto, photoz.zml, morpho.PROB_GAL, 'A119' AS cluster
FROM
idr4_dual_g AS dual_g
INNER JOIN
idr4_dual_i AS dual_i ON dual_i.ID = dual_g.ID
INNER JOIN
idr4_dual_r AS dual_r ON dual_r.ID = dual_g.ID
INNER JOIN
idr4_photoz AS photoz ON photoz.ID = dual_g.ID
INNER JOIN
idr4_star_galaxy_quasar AS morpho ON morpho.ID = dual_g.ID
WHERE
dual_g.Field = 'STRIPE82-0017' OR
dual_g.Field = 'STRIPE82-0018' OR
dual_g.Field = 'STRIPE82-0019' OR
dual_g.Field = 'STRIPE82-0020' OR
dual_g.Field = 'STRIPE82-0021' OR
dual_g.Field = 'STRIPE82-0022' OR
dual_g.Field = 'STRIPE82-0023' OR
dual_g.Field = 'STRIPE82-0025'


--------------

SELECT dual_g.ID, dual_g.RA, dual_g.DEC, dual_g.Field, dual_g.g_auto, dual_r.r_auto, dual_i.i_auto, photoz.zml, morpho.PROB_GAL, 'A147' AS cluster
FROM
idr4_dual_g AS dual_g
INNER JOIN
idr4_dual_i AS dual_i ON dual_i.ID = dual_g.ID
INNER JOIN
idr4_dual_r AS dual_r ON dual_r.ID = dual_g.ID
INNER JOIN
idr4_photoz AS photoz ON photoz.ID = dual_g.ID
INNER JOIN
idr4_star_galaxy_quasar AS morpho ON morpho.ID = dual_g.ID
WHERE
dual_g.Field = 'STRIPE82-0024' OR
dual_g.Field = 'STRIPE82-0026' OR
dual_g.Field = 'STRIPE82-0028'


-----------


SELECT dual_g.ID, dual_g.RA, dual_g.DEC, dual_g.Field, dual_g.g_auto, dual_r.r_auto, dual_i.i_auto, photoz.zml, morpho.PROB_GAL, 'A168' AS cluster
FROM
idr4_dual_g AS dual_g
INNER JOIN
idr4_dual_i AS dual_i ON dual_i.ID = dual_g.ID
INNER JOIN
idr4_dual_r AS dual_r ON dual_r.ID = dual_g.ID
INNER JOIN
idr4_photoz AS photoz ON photoz.ID = dual_g.ID
INNER JOIN
idr4_star_galaxy_quasar AS morpho ON morpho.ID = dual_g.ID
WHERE
dual_g.Field = 'STRIPE82-0027' OR
dual_g.Field = 'STRIPE82-0029' OR
dual_g.Field = 'STRIPE82-0030' OR
dual_g.Field = 'STRIPE82-0031' OR
dual_g.Field = 'STRIPE82-0032'


--------


SELECT dual_g.ID, dual_g.RA, dual_g.DEC, dual_g.Field, dual_g.g_auto, dual_r.r_auto, dual_i.i_auto, photoz.zml, morpho.PROB_GAL, 'A194' AS cluster
FROM
idr4_dual_g AS dual_g
INNER JOIN
idr4_dual_i AS dual_i ON dual_i.ID = dual_g.ID
INNER JOIN
idr4_dual_r AS dual_r ON dual_r.ID = dual_g.ID
INNER JOIN
idr4_photoz AS photoz ON photoz.ID = dual_g.ID
INNER JOIN
idr4_star_galaxy_quasar AS morpho ON morpho.ID = dual_g.ID
WHERE
dual_g.Field = 'STRIPE82-0033' OR
dual_g.Field = 'STRIPE82-0034' OR
dual_g.Field = 'STRIPE82-0035' OR
dual_g.Field = 'STRIPE82-0036' OR
dual_g.Field = 'STRIPE82-0037' OR
dual_g.Field = 'STRIPE82-0038'


---------


SELECT dual_g.ID, dual_g.RA, dual_g.DEC, dual_g.Field, dual_g.g_auto, dual_r.r_auto, dual_i.i_auto, photoz.zml, morpho.PROB_GAL, 'A2457' AS cluster
FROM
idr4_dual_g AS dual_g
INNER JOIN
idr4_dual_i AS dual_i ON dual_i.ID = dual_g.ID
INNER JOIN
idr4_dual_r AS dual_r ON dual_r.ID = dual_g.ID
INNER JOIN
idr4_photoz AS photoz ON photoz.ID = dual_g.ID
INNER JOIN
idr4_star_galaxy_quasar AS morpho ON morpho.ID = dual_g.ID
WHERE
dual_g.Field = 'STRIPE82-0139' OR
dual_g.Field = 'STRIPE82-0140' OR
dual_g.Field = 'STRIPE82-0141' OR
dual_g.Field = 'STRIPE82-0142' OR
dual_g.Field = 'STRIPE82-0143' OR
dual_g.Field = 'STRIPE82-0144'



----------



SELECT dual_g.ID, dual_g.RA, dual_g.DEC, dual_g.Field, dual_g.g_auto, dual_r.r_auto, dual_i.i_auto, photoz.zml, morpho.PROB_GAL, 'IIZw108' AS cluster
FROM
idr4_dual_g AS dual_g
INNER JOIN
idr4_dual_i AS dual_i ON dual_i.ID = dual_g.ID
INNER JOIN
idr4_dual_r AS dual_r ON dual_r.ID = dual_g.ID
INNER JOIN
idr4_photoz AS photoz ON photoz.ID = dual_g.ID
INNER JOIN
idr4_star_galaxy_quasar AS morpho ON morpho.ID = dual_g.ID
WHERE
dual_g.Field = 'STRIPE82-0110' OR
dual_g.Field = 'STRIPE82-0112' OR
dual_g.Field = 'STRIPE82-0114' OR
dual_g.Field = 'STRIPE82-0116'