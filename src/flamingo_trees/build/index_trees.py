#!/bin/env python
#
# Routines to generate depth first indexes for merger trees.
# (as in the EAGLE and Millennium databases)
#

import numpy as np
import virgo.util.match as ms


def depth_first_index(nodeIndex, descendantIndex, mBranch):
    """
    Calculate depth first indexing for the supplied tree(s).
    Progenitors are ordered by descending mBranch.
    """

    # Find array index of descendant for each halo
    desc_ptr = ms.match(descendantIndex, nodeIndex)

    # Allocate arrays to store linked lists of progenitors
    # (which will be sorted into descending order of mBranch)
    prog_ptr = -np.ones_like(desc_ptr) # points to first progenitor
    next_ptr = -np.ones_like(desc_ptr) # points to next progenitor of same descendant

    # Loop over all possible progenitor halos iprog in ascending
    # order of mBranch
    for iprog in np.argsort(mBranch):
        idesc = desc_ptr[iprog]
        if idesc >= 0:
            # Halo idesc is descendant of iprog.
            # Add it to the start of idesc's linked list.
            # Since we consider halos in ascending order of mBranch,
            # the resulting list will be in *descending* order of mBranch.
            if prog_ptr[idesc] == -1:
                # This is the first progenitor we found for halo i
                prog_ptr[idesc] = iprog
            else:
                # Replace current first progenitor with this one
                itmp = prog_ptr[idesc]
                prog_ptr[idesc] = iprog
                next_ptr[iprog] = itmp

    # Alloc. arrays for depth first indexes
    depthFirst = -np.ones_like(descendantIndex)

    # Loop over final halos
    next_index = 0
    ntot = nodeIndex.shape[0]
    for ifinal in np.where(desc_ptr<0)[0]:

        #print "Halos done: %d of %d" % (next_index, ntot)

        # Walk the tree
        ihalo = ifinal
        next_prog = [-1] # Final halo has no siblings
        while True:

            # Assign the next depth first ID to this halo
            depthFirst[ihalo] = next_index
            next_index += 1

            # Go to first progenitor while there is one, assigning
            # indexes along the way
            while prog_ptr[ihalo] >= 0:

                ihalo = prog_ptr[ihalo]           # go to first progenitor
                next_prog.append(next_ptr[ihalo]) # push sibling index to stack
                depthFirst[ihalo] = next_index    # assign next index
                next_index += 1

            # Now we're at the end of the branch, go back up
            # until we find a sibling halo
            while desc_ptr[ihalo] >= 0:
                isibling = next_prog.pop()
                if isibling >= 0:
                    # This one has a sibling, so visit it
                    ihalo = isibling
                    # Push sibling's sibling (if any) to the stack
                    next_prog.append(next_ptr[ihalo])
                    break
                else:
                    # No sibling, go to descendant
                    ihalo = desc_ptr[ihalo]

            # If we get back to where we started, we're done
            if ihalo == ifinal:
                break

    # Alloc. arrays for last progenitor and end of main branch
    endMainBranch  = depthFirst.copy()
    lastProgenitor = depthFirst.copy()

    # Loop over halos in descending order of depth first ID
    for iprog in np.argsort(-depthFirst):
        idesc = desc_ptr[iprog]
        if idesc >= 0:
            # Update maximum ID of any progenitor of this descendant
            lastProgenitor[idesc] = max(lastProgenitor[iprog], lastProgenitor[idesc])
            # If we're on the main branch, updated end of main branch ID
            if prog_ptr[idesc] == iprog:
                endMainBranch[idesc] = max(endMainBranch[iprog], endMainBranch[idesc])

    # All array elements should have been set
    assert np.all(endMainBranch  >= 0)
    assert np.all(lastProgenitor >= 0)
    assert np.all(depthFirst     >= 0)

    # Should always have lastProgenitor >= endMainBranch >= depthFirst
    assert np.all(lastProgenitor >= endMainBranch)
    assert np.all(endMainBranch  >= depthFirst)

    # Return array with the indexes
    return depthFirst, endMainBranch, lastProgenitor
