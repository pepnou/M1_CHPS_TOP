/********************  HEADERS  *********************/
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "lbm_comm.h"

/*******************  FUNCTION  *********************/
int lbm_helper_pgcd(int a, int b)
{
	printf("%d %d\n", a, b);
	int c;
	while(b!=0)
	{
		c = a % b;
		a = b;
		b = c;
	}
	return a;
}

/*******************  FUNCTION  *********************/
/**
 * Affiche la configuation du lbm_comm pour un rank donné
 * @param mesh_comm Configuration à afficher
**/
void  lbm_comm_print( lbm_comm_t *mesh_comm )
{
	int rank ;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	printf( " RANK %d ( LEFT %d RIGHT %d TOP %d BOTTOM %d CORNER %d, %d, %d, %d ) ( POSITION %d %d ) (WH %d %d ) \n", rank,
									    mesh_comm->left_id,
									    mesh_comm->right_id,
										mesh_comm->top_id,
									    mesh_comm->bottom_id,
										mesh_comm->corner_id[0],
		 								mesh_comm->corner_id[1],
		 								mesh_comm->corner_id[2],
		 		 						mesh_comm->corner_id[3],
									    mesh_comm->x,
									    mesh_comm->y,
									    mesh_comm->width,
									    mesh_comm->height );
}

/*******************  FUNCTION  *********************/
int helper_get_rank_id(int nb_x,int nb_y,int rank_x,int rank_y)
{
	if (rank_x < 0 || rank_x >= nb_x)
		return -1;
	else if (rank_y < 0 || rank_y >= nb_y)
		return -1;
	else
		return (rank_x + rank_y * nb_x);
}

/*******************  FUNCTION  *********************/
/**
 * Initialise un lbm_comm :
 * - Voisins
 * - Taille du maillage local
 * - Position relative
 * @param mesh_comm MeshComm à initialiser
 * @param rank Rank demandant l'initalisation
 * @param comm_size Taille totale du communicateur
 * @param width largeur du maillage
 * @param height hauteur du maillage
**/
void lbm_comm_init( lbm_comm_t * mesh_comm, int rank, int comm_size, int width, int height )
{
	//vars
	int nb_x;
	int nb_y;
	int rank_x;
	int rank_y;

	//compute splitting
	//nb_y = lbm_helper_pgcd(comm_size,width);
	//nb_x = comm_size / nb_y;
	nb_x = lbm_helper_pgcd(comm_size,width);
	nb_y = comm_size / nb_x;

	//check
	printf("%d %d\n", nb_x, nb_y);
	assert(nb_x * nb_y == comm_size);
	if (height % nb_y != 0)
		fatal("Can't get a 2D cut for current problem size and number of processes.");
	if (width % nb_x != 0)
	 	fatal("Can't get a 2D cut for current problem size and number of processes.");

	//calc current rank position (ID)
	rank_x = rank % nb_x;
	rank_y = rank / nb_x;

	//setup nb
	mesh_comm->nb_x = nb_x;
	mesh_comm->nb_y = nb_y;

	//setup size (+2 for ghost cells on border)
	mesh_comm->width = width / nb_x + 2;
	mesh_comm->height = height / nb_y + 2;

	//setup position
	mesh_comm->x = rank_x * width / nb_x;
	mesh_comm->y = rank_y * height / nb_y;
	
	// Compute neighbour nodes id
	mesh_comm->left_id  = helper_get_rank_id(nb_x,nb_y,rank_x - 1,rank_y);
	mesh_comm->right_id = helper_get_rank_id(nb_x,nb_y,rank_x + 1,rank_y);
	mesh_comm->top_id = helper_get_rank_id(nb_x,nb_y,rank_x,rank_y - 1);
	mesh_comm->bottom_id = helper_get_rank_id(nb_x,nb_y,rank_x,rank_y + 1);
	mesh_comm->corner_id[CORNER_TOP_LEFT] = helper_get_rank_id(nb_x,nb_y,rank_x - 1,rank_y - 1);
	mesh_comm->corner_id[CORNER_TOP_RIGHT] = helper_get_rank_id(nb_x,nb_y,rank_x + 1,rank_y - 1);
	mesh_comm->corner_id[CORNER_BOTTOM_LEFT] = helper_get_rank_id(nb_x,nb_y,rank_x - 1,rank_y + 1);
	mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = helper_get_rank_id(nb_x,nb_y,rank_x + 1,rank_y + 1);

	//if more than 1 on y, need transmission buffer
	if (nb_y > 1)
	{
		mesh_comm->buffer = malloc(sizeof(double) * DIRECTIONS * width / nb_x);
	} else {
		mesh_comm->buffer = NULL;
	}


	//datatype init
	MPI_Datatype g,d,h,b;

	int blcklen1[] = {1,2};
	int blcklen2[] = {1,1,1};

	int indexg[] = {3,6};
	int indexd[] = {1,5,8};
	int indexh[] = {2,5};
	int indexb[] = {4,7};

	MPI_Type_indexed(2,blcklen1,indexg,MPI_DOUBLE,&g);
	MPI_Type_indexed(3,blcklen2,indexd,MPI_DOUBLE,&d);
	MPI_Type_indexed(2,blcklen1,indexh,MPI_DOUBLE,&h);
	MPI_Type_indexed(2,blcklen1,indexb,MPI_DOUBLE,&b);

	MPI_Type_commit(&g);
	MPI_Type_commit(&d);
	MPI_Type_commit(&h);
	MPI_Type_commit(&b);

	MPI_Type_create_hvector( height / nb_y, 1,9 * sizeof(double), g, &(mesh_comm->left));
	MPI_Type_create_hvector( height / nb_y, 1,9 * sizeof(double), d, &(mesh_comm->right));
	MPI_Type_create_hvector( width / nb_x, 1,9 * sizeof(double) * (height / nb_y), h, &(mesh_comm->up));
	MPI_Type_create_hvector( width / nb_x, 1,9 * sizeof(double) * (height / nb_y), b, &(mesh_comm->down));

	MPI_Type_commit(&(mesh_comm->left));
	MPI_Type_commit(&(mesh_comm->right));
	MPI_Type_commit(&(mesh_comm->up));
	MPI_Type_commit(&(mesh_comm->down));


	//if debug print comm
	#ifndef NDEBUG
	lbm_comm_print( mesh_comm );
	#endif
}


/*******************  FUNCTION  *********************/
/**
 * Libere un lbm_comm
 * @param mesh_comm MeshComm à liberer
**/
void lbm_comm_release( lbm_comm_t * mesh_comm )
{
	mesh_comm->x = 0;
	mesh_comm->y = 0;
	mesh_comm->width = 0;
	mesh_comm->height = 0;
	mesh_comm->right_id = -1;
	mesh_comm->left_id = -1;
	if (mesh_comm->buffer != NULL)
		free(mesh_comm->buffer);
	mesh_comm->buffer = NULL;
}

/*******************  FUNCTION  *********************/
/**
 * Debut de communications asynchrones
 * @param mesh_comm MeshComm à utiliser
 * @param mesh_to_process Mesh a utiliser lors de l'échange des mailles fantomes
**/
void lbm_comm_sync_ghosts_horizontal( lbm_comm_t * mesh, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int x, MPI_Datatype datatype)
{
	//vars
	//MPI_Status status;

	//if target is -1, no comm
	if (target_rank == -1)
		return;

	//int y;

	switch (comm_type)
	{
		case COMM_SEND:
			MPI_Isend( &Mesh_get_col( mesh_to_process, x )[0], 1, datatype, target_rank, 0, MPI_COMM_WORLD, &(mesh->requests[mesh->request_count]));
			break;
		case COMM_RECV:
			MPI_Irecv(  &Mesh_get_col( mesh_to_process, x )[0], 1, datatype, target_rank, 0, MPI_COMM_WORLD, &(mesh->requests[mesh->request_count]));
			break;
		default:
			fatal("Unknown type of communication.");
	}

	mesh->request_count += 1;
}

/*******************  FUNCTION  *********************/
/**
 * Debut de communications asynchrones
 * @param mesh_comm MeshComm à utiliser
 * @param mesh_to_process Mesh a utiliser lors de l'échange des mailles fantomes
**/
void lbm_comm_sync_ghosts_diagonal(lbm_comm_t * mesh, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int x ,int y)
{
	//vars
	//MPI_Status status;

	//if target is -1, no comm
	if (target_rank == -1)
		return;

	switch (comm_type)
	{
		case COMM_SEND:
			MPI_Isend( Mesh_get_cell( mesh_to_process, x, y ), DIRECTIONS, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &(mesh->requests[mesh->request_count]));
			break;
		case COMM_RECV:
			MPI_Irecv( Mesh_get_cell( mesh_to_process, x, y ), DIRECTIONS, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &(mesh->requests[mesh->request_count]));
			break;
		default:
			fatal("Unknown type of communication.");
	}

	mesh->request_count += 1;
}


/*******************  FUNCTION  *********************/
/**
 * Debut de communications asynchrones
 * @param mesh_comm MeshComm à utiliser
 * @param mesh_to_process Mesh a utiliser lors de l'échange des mailles fantomes
**/
void lbm_comm_sync_ghosts_vertical( lbm_comm_t * mesh, Mesh *mesh_to_process, lbm_comm_type_t comm_type, int target_rank, int y, MPI_Datatype datatype)
{
	//vars
	//MPI_Status status;
	//int x/*, k*/;

	//if target is -1, no comm
	if (target_rank == -1)
		return;

	switch (comm_type)
	{
		case COMM_SEND:
			MPI_Isend( Mesh_get_cell(mesh_to_process, 1, y), 1, datatype, target_rank, 0, MPI_COMM_WORLD, &(mesh->requests[mesh->request_count]));
			break;
		case COMM_RECV:
			MPI_Irecv( Mesh_get_cell(mesh_to_process, 1, y), 1, datatype, target_rank, 0, MPI_COMM_WORLD, &(mesh->requests[mesh->request_count]));
			break;
		default:
			fatal("Unknown type of communication.");
	}

	mesh->request_count += 1;
}

/*******************  FUNCTION  *********************/
void lbm_comm_ghost_exchange(lbm_comm_t * mesh, Mesh *mesh_to_process )
{
	//vars
	int rank;
	mesh->request_count = 0;

	//get rank
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	if(rank % 2)
	{
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->right_id,mesh->width - 1, mesh->right);
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->left_id, 1, mesh->right);

		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->left_id, 0, mesh->left);
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->right_id,mesh->width - 2, mesh->left);
	}
	else
	{
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->left_id, 1, mesh->right);
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->right_id,mesh->width - 1, mesh->right);
		
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_RECV,mesh->right_id,mesh->width - 2, mesh->left);
		lbm_comm_sync_ghosts_horizontal(mesh,mesh_to_process,COMM_SEND,mesh->left_id, 0, mesh->left);
	}

	if(rank % 2)
	{
		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_SEND,mesh->bottom_id,mesh->height - 1, mesh->down);
		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_RECV,mesh->top_id, 1, mesh->down);

		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_SEND,mesh->top_id, 0, mesh->up);
		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_RECV,mesh->bottom_id,mesh->height - 2, mesh->up);
	}
	else
	{
		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_RECV,mesh->top_id, 1, mesh->down);
		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_SEND,mesh->bottom_id,mesh->height - 1, mesh->down);

		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_RECV,mesh->bottom_id,mesh->height - 2, mesh->up);
		lbm_comm_sync_ghosts_vertical(mesh,mesh_to_process,COMM_SEND,mesh->top_id, 0, mesh->up);
	}

	//MPI_Request rqsts[8];
	//int rqst_count = 0;

	//top left
	lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_TOP_LEFT],1,1);
	lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_BOTTOM_RIGHT],mesh->width - 1,mesh->height - 1);

	//bottom left
	lbm_comm_sync_ghosts_diagonal( mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_BOTTOM_LEFT],1,mesh->height - 2);
	lbm_comm_sync_ghosts_diagonal( mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_TOP_RIGHT],mesh->width - 1,0);

	//top right
	lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_TOP_RIGHT],mesh->width - 2,1);
	lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_BOTTOM_LEFT],0,mesh->height - 1);

	//bottom right
	lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_SEND,mesh->corner_id[CORNER_BOTTOM_RIGHT],mesh->width - 2,mesh->height - 2);
	lbm_comm_sync_ghosts_diagonal(mesh,mesh_to_process,COMM_RECV,mesh->corner_id[CORNER_TOP_LEFT],0,0);

	MPI_Waitall(mesh->request_count, mesh->requests, MPI_STATUSES_IGNORE);

	//wait for nothing to finish, not VERY important, do remove.
	//FLUSH_INOUT();
}

/*******************  FUNCTION  *********************/
/**
 * Rendu du mesh en effectuant une réduction a 0
 * @param mesh_comm MeshComm à utiliser
 * @param temp Mesh a utiliser pour stocker les segments
**/
void save_frame_all_domain( FILE * fp, Mesh *source_mesh, Mesh *temp )
{
	//vars
	int i = 0;
	int comm_size, rank ;
	MPI_Status status;

	//get rank and comm size
	MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	/* If whe have more than one process */
	if( 1 < comm_size )
	{
		if( rank == 0 )
		{
			/* Rank 0 renders its local Mesh */
			save_frame(fp,source_mesh);
			/* Rank 0 receives & render other processes meshes */
			for( i = 1 ; i < comm_size ; i++ )
			{
				MPI_Recv( temp->cells, source_mesh->width  * source_mesh->height * DIRECTIONS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status );
				save_frame(fp,temp);
			}
		} else {
			/* All other ranks send their local mesh */
			MPI_Send( source_mesh->cells, source_mesh->width * source_mesh->height * DIRECTIONS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
		}
	} else {
		/* Only 0 renders its local mesh */
		save_frame(fp,source_mesh);
	}

}

