/*
 * Copyright (c) 2011-2012, fortiss GmbH.
 * Licensed under the Apache License, Version 2.0.
 *
 * Use, modification and distribution are subject to the terms specified
 * in the accompanying license file LICENSE.txt located at the root directory
 * of this software distribution. A copy is available at
 * http://chromosome.fortiss.org/.
 *
 * This file is part of CHROMOSOME.
 *
 * $Id$
 */

/**
 * \file
 *         Math functions (architecture specific part: generic C stub).
 *
 * \author
 *         Simon Barner <barner@fortiss.org>
 *         Michael Geisinger <geisinger@fortiss.org>
 */

#ifndef XME_HAL_MATH_ARCH_H
#define XME_HAL_MATH_ARCH_H

/**
 * \defgroup hal_math Math
 *
 * \brief  Math functions.
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "stdint.h"

/******************************************************************************/
/***   Defines                                                              ***/
/******************************************************************************/

/******************************************************************************/
/***   Prototypes                                                           ***/
/******************************************************************************/
/**
 * \brief  Returns the least power of 2 greater than or equal to the given value.
 *
 * \note   Note that for x=0 and for x>2147483648 this returns 0!
 *         Implemented after http://www.flipcode.com/archives/Some_More_Power_Of_2_Utility_Functions.shtml.
 *
 * \param  x Unsigned input number.
 * \return Least power of 2 greater than or equal to the given value.
 */
uint32_t
xme_hal_math_ceilPowerOfTwo(uint32_t x);

/**
 * \brief  Returns the greatest power of 2 less than or equal to the given value.
 *
 * \note   Note that for x=0 this returns 0!
 *         Implemented after http://www.flipcode.com/archives/Some_More_Power_Of_2_Utility_Functions.shtml.
 *
 * \param  x Unsigned input number.
 * \return Greatest power of 2 less than or equal to the given value.
 */
uint32_t
xme_hal_math_floorPowerOfTwo(uint32_t x);

#endif // #ifndef XME_HAL_MATH_ARCH_H
