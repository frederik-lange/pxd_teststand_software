/**
 * @file
 * Ethernet Interface Skeleton
 *
 */

/*
 * Copyright (c) 2001-2004 Swedish Institute of Computer Science.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 * This file is part of the lwIP TCP/IP stack.
 *
 * Author: Adam Dunkels <adam@sics.se>
 *
 */

/*
 * This file is a skeleton for developing Ethernet network interface
 * drivers for lwIP. Add code to the low_level functions and do a
 * search-and-replace for the word "ethernetif" to replace it with
 * something that better describes your network interface.
 */

// Managed by xme_build_option()
#include "lwip/opt.h"
#include "bsp/bsp_opt.h"
#include "network_settings.h"

#include "lwip/def.h"
#include "lwip/mem.h"
#include "lwip/pbuf.h"
#include "lwip/sys.h"
#include "lwip/stats.h"
#include "lwip/snmp.h"
#include "netif/etharp.h"
#include "netif/ppp_oe.h"
#include "lwip/err.h"
#include "ethernetif.h"

#include "stm32_eth.h"

#include <string.h>

#define netifMTU                                (1500)
#define netifINTERFACE_TASK_STACK_SIZE		( 350 )
#define netifINTERFACE_TASK_PRIORITY		( configMAX_PRIORITIES - 1 )
#define netifGUARD_BLOCK_TIME			( 250 )
/* The time to block waiting for input. */
#define emacBLOCK_TIME_WAITING_FOR_INPUT	( ( portTickType ) 100 )

/* Define those to better describe your network interface. */
#define IFNAME0 's'
#define IFNAME1 't'


#define  ETH_DMARxDesc_FrameLengthShift           16
#define  ETH_ERROR              ((u32)0)
#define  ETH_SUCCESS            ((u32)1)

/**
 * Number of pbufs supported in low-level tx/rx pbuf queue.
 *
 */
#ifndef STM32_NUM_PBUF_QUEUE
#define STM32_NUM_PBUF_QUEUE    20
#endif

#define HWREG(x)	(*((volatile unsigned long *)(x)))

#define PBUF_QUEUE_EMPTY(q) (((q)->qwrite == (q)->qread) ? 1 : 0)


/**
 * Sanity Check:  This interface driver will NOT work if the following defines
 * are incorrect.
 *
 */
#if (PBUF_LINK_HLEN != 14)
#error "PBUF_LINK_HLEN must be 14 for this interface driver!"
#endif
#if (ETH_PAD_SIZE != 0)
#error "ETH_PAD_SIZE must be 0 for this interface driver!"
#endif
#if (!SYS_LIGHTWEIGHT_PROT)
#error "SYS_LIGHTWEIGHT_PROT must be enabled for this interface driver!"
#endif

/* TCP and ARP timeouts */
volatile int tcp_end_time, arp_end_time;

static void ethernetif_input( void * pvParameters );
void vEMACWaitForInput( void );
static void arp_timer(void *arg);

/* Helper struct to hold a queue of pbufs for transmit and receive. */
struct pbufq {
  struct pbuf *pbuf[STM32_NUM_PBUF_QUEUE];
  unsigned long qwrite;
  unsigned long qread;
  unsigned long overflow;
};

/**
 * Helper struct to hold private data used to operate your ethernet interface.
 * Keeping the ethernet address of the MAC in this struct is not necessary
 * as it is already kept in the struct netif.
 * But this is only an example, anyway...
 */
struct ethernetif
{
  struct eth_addr *ethaddr;
  /* Add whatever per-interface state that is needed here. */
  int unused;
  struct pbufq txq;
};

xSemaphoreHandle s_xSemaphore = NULL;



static struct netif *s_pxNetIf = NULL;

#define ETH_RXBUFNB        4
#define ETH_TXBUFNB        2

uint8_t MACaddr[6];
ETH_DMADESCTypeDef  DMARxDscrTab[ETH_RXBUFNB], DMATxDscrTab[ETH_TXBUFNB];/* Ethernet Rx & Tx DMA Descriptors */
uint8_t Rx_Buff[ETH_RXBUFNB][ETH_MAX_PACKET_SIZE], Tx_Buff[ETH_TXBUFNB][ETH_MAX_PACKET_SIZE];/* Ethernet buffers */

ETH_DMADESCTypeDef  *DMATxDesc = DMATxDscrTab;
extern ETH_DMADESCTypeDef  *DMATxDescToSet;
extern ETH_DMADESCTypeDef  *DMARxDescToGet;

typedef struct{
u32 length;
u32 buffer;
ETH_DMADESCTypeDef *descriptor;
}FrameTypeDef;

FrameTypeDef ETH_RxPkt_ChainMode(void);
u32 ETH_GetCurrentTxBuffer(void);
u32 ETH_TxPkt_ChainMode(u16 FrameLength);

/**
 * Pop a pbuf packet from a pbuf packet queue
 *
 * @param q is the packet queue from which to pop the pbuf.
 *
 * @return pointer to pbuf packet if available, NULL otherswise.
 */
static struct pbuf *
dequeue_packet(struct pbufq *q)
{
  struct pbuf *pBuf;
  SYS_ARCH_DECL_PROTECT(lev);

  /**
   * This entire function must run within a "critical section" to preserve
   * the integrity of the transmit pbuf queue.
   *
   */
  SYS_ARCH_PROTECT(lev);

  if(PBUF_QUEUE_EMPTY(q)) {
    /* Return a NULL pointer if the queue is empty. */
    pBuf = (struct pbuf *)NULL;
  }
  else {
    /**
     * The queue is not empty so return the next frame from it
     * and adjust the read pointer accordingly.
     *
     */
    pBuf = q->pbuf[q->qread];
    q->qread = ((q->qread + 1) % STM32_NUM_PBUF_QUEUE);
  }

  /* Return to prior interrupt state and return the pbuf pointer. */
  SYS_ARCH_UNPROTECT(lev);
  return(pBuf);
}
/**
 * In this function, the hardware should be initialized.
 * Called from ethernetif_init().
 *
 * @param netif the already initialized lwip network interface structure
 *        for this ethernetif
 */
static void
low_level_init(struct netif *netif)
{
	/* set MAC hardware address length */
	netif->hwaddr_len = ETHARP_HWADDR_LEN;
	/* set MAC hardware address */

	// TODO: Redundant?
	netif->hwaddr[0] =  MAC_ADDR0;
  	netif->hwaddr[1] =  MAC_ADDR1;
  	netif->hwaddr[2] =  MAC_ADDR2;
  	netif->hwaddr[3] =  MAC_ADDR3;
  	netif->hwaddr[4] =  MAC_ADDR4;
  	netif->hwaddr[5] =  MAC_ADDR5;
  
 	 /* maximum transfer unit */
  	netif->mtu = 1500;

  	/* Accept broadcast address and ARP traffic */
    netif->flags = NETIF_FLAG_BROADCAST | NETIF_FLAG_ETHARP | NETIF_FLAG_IGMP | NETIF_FLAG_LINK_UP;
 	
  	s_pxNetIf =netif;
  
  	if (s_xSemaphore == NULL)
  	{
    	vSemaphoreCreateBinary(s_xSemaphore);
    	xSemaphoreTake(s_xSemaphore, 0);
  	}

  	/* initialize MAC address in ETH_MAC*/ 
  	ETH_MACAddressConfig(ETH_MAC_Address0, netif->hwaddr); 
  
  	/* Initialize Tx Descriptors list: Chain Mode */
  	ETH_DMATxDescChainInit(DMATxDscrTab, &Tx_Buff[0][0], ETH_TXBUFNB);
  	/* Initialize Rx Descriptors list: Chain Mode  */
  	ETH_DMARxDescChainInit(DMARxDscrTab, &Rx_Buff[0][0], ETH_RXBUFNB);

  	/* Enable Ethernet Rx interrrupt */
  	{ 
  		int i;
    	for(i=0; i<ETH_RXBUFNB; i++)
    	{
      		ETH_DMARxDescReceiveITConfig(&DMARxDscrTab[i], ENABLE);
    	}
  	}

	#ifdef CHECKSUM_BY_HARDWARE
  	/* Enable the checksum insertion for the Tx frames */
  	{
  		int i;
    	for(i=0; i<ETH_TXBUFNB; i++)
    	{
      		ETH_DMATxDescChecksumInsertionConfig(&DMATxDscrTab[i], ETH_DMATxDesc_ChecksumTCPUDPICMPFull);
    	}
  	}
	#endif

  	/* Enable MAC and DMA transmission and reception */
  	ETH_Start();
  
  	// TODO
  	/* create the task that handles the ETH_MAC */
  	//xTaskCreate(ethernetif_input, (signed char*) "ETH_INT", netifINTERFACE_TASK_STACK_SIZE, NULL,
    //          netifINTERFACE_TASK_PRIORITY,NULL);

}

/**
 * This function should do the actual transmission of the packet. The packet is
 * contained in the pbuf that is passed to the function. This pbuf
 * might be chained.
 *
 * @param netif the lwip network interface structure for this ethernetif
 * @param p the MAC packet to send (e.g. IP packet including MAC addresses and type)
 * @return ERR_OK if the packet could be sent
 *         an err_t value if the packet couldn't be sent
 *
 * @note Returning ERR_MEM here if a DMA queue of your MAC is full can lead to
 *       strange results. You might consider waiting for space in the DMA queue
 *       to become availale since the stack doesn't retry to send a packet
 *       dropped because of memory failure (except for the TCP timers).
 */

static err_t
low_level_output(struct netif *netif, struct pbuf *p)
{
	static xSemaphoreHandle xTxSemaphore = NULL;
  	struct pbuf *q;
  	int l = 0;
  	u8 *buffer ;
  
  	if (xTxSemaphore == NULL)
  	{
  		vSemaphoreCreateBinary (xTxSemaphore);
  	} 
    
  	if (xSemaphoreTake(xTxSemaphore, netifGUARD_BLOCK_TIME))
  	{
  		u8 *buffer =  (u8 *)ETH_GetCurrentTxBuffer();
  		for(q = p; q != NULL; q = q->next) 
  		{
  			memcpy((u8_t*)&buffer[l], q->payload, q->len);
  			l = l + q->len;
  		}
  	}
  	xSemaphoreGive(xTxSemaphore);
  	ETH_TxPkt_ChainMode(l);

 	 return ERR_OK;
}




/**
 * Should allocate a pbuf and transfer the bytes of the incoming
 * packet from the interface into the pbuf.
 *
 * @param netif the lwip network interface structure for this ethernetif
 * @return a pbuf filled with the received packet (including MAC header)
 *         NULL on memory error
 */
static struct pbuf *
low_level_input(struct netif *netif)
{
 static xSemaphoreHandle xRxSemaphore = NULL;


  struct pbuf *p, *q;
  u16_t len;
  int l =0;
  FrameTypeDef frame;
  u8 *buffer;
  
  p = NULL;

 if( xRxSemaphore ==NULL)
 	{
 	vSemaphoreCreateBinary(xRxSemaphore);
 	}
 /* access to emac is guarded using a semphore */

 if (xSemaphoreTake(xRxSemaphore, netifGUARD_BLOCK_TIME))
 	{
          
          
          
 
  frame = ETH_RxPkt_ChainMode();
  /* Obtain the size of the packet and put it into the "len"
     variable. */
  len = frame.length;
  
  if (len)
  {
  buffer = (u8 *)frame.buffer;

  /* We allocate a pbuf chain of pbufs from the pool. */
  p = pbuf_alloc(PBUF_RAW, len, PBUF_POOL);
 
  if (p != NULL)
  {
    for (q = p; q != NULL; q = q->next)
    {
	  memcpy((u8_t*)q->payload, (u8_t*)&buffer[l], q->len);
      l = l + q->len;
    }    
  
  }
  /* Set Own bit of the Rx descriptor Status: gives the buffer back to ETHERNET DMA */
  frame.descriptor->Status = ETH_DMARxDesc_OWN; 
 
  /* When Rx Buffer unavailable flag is set: clear it and resume reception */
  if ((ETH->DMASR & ETH_DMASR_RBUS) != (u32)RESET)  
  {
    /* Clear RBUS ETHERNET DMA flag */
    ETH->DMASR = ETH_DMASR_RBUS;
    /* Resume DMA reception */
    ETH->DMARPDR = 0;
  }
  }
 }
 xSemaphoreGive(xRxSemaphore);
  return p;
}


/**
 * This function should be called when a packet is ready to be read
 * from the interface. It uses the function low_level_input() that
 * should handle the actual reception of bytes from the network
 * interface. Then the type of the received packet is determined and
 * the appropriate input function is called.
 *
 * @param netif the lwip network interface structure for this ethernetif
 */
static void ethernetif_input( void * pvParameters )
{
  struct ethernetif *ethernetif;
  struct eth_hdr *ethhdr;
  struct pbuf *p;

//	for( ;; )
	{
		do
		{
			ethernetif = s_pxNetIf->state;
			
			/* move received packet into a new pbuf */
			p = low_level_input( s_pxNetIf );
			
			if( p == NULL )
			{
				/* No packet could be read.  Wait a for an interrupt to tell us
				there is more data available. */
				vEMACWaitForInput();
			}
		
		} while( p == NULL );

		/* points to packet payload, which starts with an Ethernet header */
		ethhdr = p->payload;

		#if LINK_STATS
			lwip_stats.link.recv++;
		#endif /* LINK_STATS */

		ethhdr = p->payload;

		switch (htons(ethhdr->type))
		{
			/* IP packet? */
		case 256:
			case ETHTYPE_IP:
				
                          /* full packet send to tcpip_thread to process */
			if (s_pxNetIf->input(p, s_pxNetIf) != ERR_OK)
			{
				LWIP_DEBUGF(NETIF_DEBUG, ("ethernetif_input: IP input error\n"));
				pbuf_free(p);
				p = NULL;
			}
			break;
      
      case ETHTYPE_ARP:                    
			  /* pass p to ARP module  */
    	  	  ethernet_input(p, s_pxNetIf);
			  break;
				
			default:
				  pbuf_free(p);
				  p = NULL;
				  break;
		}
	}
}

// This is implementation of an ethernetif_input similar to the one for the stellaris
void
stm32_ethernetif_input(struct netif *netif)
{
  struct ethernetif *ethernetif;
  struct eth_hdr *ethhdr;
  struct pbuf *p;
  int i,res;
  unsigned char *input_string;

  /* setup pointer to the if state data */
  ethernetif = netif->state;
  /**
   * Process the transmit and receive queues as long as there is receive
   * data available
   *
   */
  p = low_level_input(netif);
  while(p != NULL) {
	/* process the packet */
#if NO_SYS
	if(ethernet_input(p, netif)!=ERR_OK) {
#else
	if(tcpip_input(p, netif)!=ERR_OK) {
#endif
	  /* drop the packet */
	  LWIP_DEBUGF(NETIF_DEBUG, ("stellarisif_input: input error\n"));
	  pbuf_free(p);
	  /* Adjust the link statistics */
	  LINK_STATS_INC(link.memerr);
	  LINK_STATS_INC(link.drop);
}

	/* Check if TX fifo is empty and packet available */
	if((HWREG(ETH_BASE + MAC_O_TR) & MAC_TR_NEWTX) == 0) {
	  p = dequeue_packet(&ethernetif->txq);
	  if(p != NULL) {
		  low_level_output(netif, p);
	  }
	}

	/* Read another packet from the RX fifo */
	p = low_level_input(netif);
  }

  /* One more check of the transmit queue/fifo */
  if((HWREG(ETH_BASE + MAC_O_TR) & MAC_TR_NEWTX) == 0) {
	p = dequeue_packet(&ethernetif->txq);
	if(p != NULL) {
		low_level_output(netif, p);
	}
  }
}

/**
 * Should be called at the beginning of the program to set up the
 * network interface. It calls the function low_level_init() to do the
 * actual setup of the hardware.
 *
 * This function should be passed as a parameter to netif_add().
 *
 * @param netif the lwip network interface structure for this ethernetif
 * @return ERR_OK if the loopif is initialized
 *         ERR_MEM if private data couldn't be allocated
 *         any other err_t on error
 */
err_t
ethernetif_init(struct netif *netif)
{
  struct ethernetif *ethernetif;

  LWIP_ASSERT("netif != NULL", (netif != NULL));

  ethernetif = mem_malloc(sizeof(struct ethernetif));
  if (ethernetif == NULL)
  {
    LWIP_DEBUGF(NETIF_DEBUG, ("ethernetif_init: out of memory\n"));
    return ERR_MEM;
  }

#if LWIP_NETIF_HOSTNAME
  /* Initialize interface hostname */
  netif->hostname = "lwip";
#endif /* LWIP_NETIF_HOSTNAME */

  /*
   * Initialize the snmp variables and counters inside the struct netif.
   * The last argument should be replaced with your link speed, in units
   * of bits per second.
   */
  NETIF_INIT_SNMP(netif, snmp_ifType_ethernet_csmacd, 100000000);

  netif->state = ethernetif;
  netif->name[0] = IFNAME0;
  netif->name[1] = IFNAME1;
  /* We directly use etharp_output() here to save a function call.
   * You can instead declare your own function an call etharp_output()
   * from it if you have to do some checks before sending (e.g. if link
   * is available...) */
  netif->output = etharp_output;
  netif->linkoutput = low_level_output;

  ethernetif->ethaddr = (struct eth_addr *)&(netif->hwaddr[0]);

  /* initialize the hardware */
  low_level_init(netif);
  
  return ERR_OK;
}

/*******************************************************************************
* Function Name  : ETH_RxPkt_ChainMode
* Description    : Receives a packet.
* Input          : None
* Output         : None
* Return         : frame: farme size and location
*******************************************************************************/
FrameTypeDef ETH_RxPkt_ChainMode(void)
{ 
  u32 framelength = 0;
  FrameTypeDef frame = {0,0}; 

  /* Check if the descriptor is owned by the ETHERNET DMA (when set) or CPU (when reset) */
  if((DMARxDescToGet->Status & ETH_DMARxDesc_OWN) != (u32)RESET)
  {	
	frame.length = ETH_ERROR;

    if ((ETH->DMASR & ETH_DMASR_RBUS) != (u32)RESET)  
    {
      /* Clear RBUS ETHERNET DMA flag */
      ETH->DMASR = ETH_DMASR_RBUS;
      /* Resume DMA reception */
      ETH->DMARPDR = 0;
    }

	/* Return error: OWN bit set */
    return frame; 
  }
  
  if(((DMARxDescToGet->Status & ETH_DMARxDesc_ES) == (u32)RESET) && 
     ((DMARxDescToGet->Status & ETH_DMARxDesc_LS) != (u32)RESET) &&  
     ((DMARxDescToGet->Status & ETH_DMARxDesc_FS) != (u32)RESET))  
  {      
    /* Get the Frame Length of the received packet: substruct 4 bytes of the CRC */
    framelength = ((DMARxDescToGet->Status & ETH_DMARxDesc_FL) >> ETH_DMARxDesc_FrameLengthShift) - 4;
	
	/* Get the addrees of the actual buffer */
	frame.buffer = DMARxDescToGet->Buffer1Addr;	
  }
  else
  {
    /* Return ERROR */
    framelength = ETH_ERROR;
  }

  frame.length = framelength;


  frame.descriptor = DMARxDescToGet;
  
  /* Update the ETHERNET DMA global Rx descriptor with next Rx decriptor */      
  /* Chained Mode */    
  /* Selects the next DMA Rx descriptor list for next buffer to read */ 
  DMARxDescToGet = (ETH_DMADESCTypeDef*) (DMARxDescToGet->Buffer2NextDescAddr);    
  
  /* Return Frame */
  return (frame);  
}

/*******************************************************************************
* Function Name  : ETH_TxPkt_ChainMode
* Description    : Transmits a packet, from application buffer, pointed by ppkt.
* Input          : - FrameLength: Tx Packet size.
* Output         : None
* Return         : ETH_ERROR: in case of Tx desc owned by DMA
*                  ETH_SUCCESS: for correct transmission
*******************************************************************************/
u32 ETH_TxPkt_ChainMode(u16 FrameLength)
{   
  /* Check if the descriptor is owned by the ETHERNET DMA (when set) or CPU (when reset) */
  if((DMATxDescToSet->Status & ETH_DMATxDesc_OWN) != (u32)RESET)
  {  
	/* Return ERROR: OWN bit set */
    return ETH_ERROR;
  }
        
  /* Setting the Frame Length: bits[12:0] */
  DMATxDescToSet->ControlBufferSize = (FrameLength & ETH_DMATxDesc_TBS1);

  /* Setting the last segment and first segment bits (in this case a frame is transmitted in one descriptor) */    
  DMATxDescToSet->Status |= ETH_DMATxDesc_LS | ETH_DMATxDesc_FS;

  /* Set Own bit of the Tx descriptor Status: gives the buffer back to ETHERNET DMA */
  DMATxDescToSet->Status |= ETH_DMATxDesc_OWN;

  /* When Tx Buffer unavailable flag is set: clear it and resume transmission */
  if ((ETH->DMASR & ETH_DMASR_TBUS) != (u32)RESET)
  {
    /* Clear TBUS ETHERNET DMA flag */
    ETH->DMASR = ETH_DMASR_TBUS;
    /* Resume DMA transmission*/
    ETH->DMATPDR = 0;
  }
  
  /* Update the ETHERNET DMA global Tx descriptor with next Tx decriptor */  
  /* Chained Mode */
  /* Selects the next DMA Tx descriptor list for next buffer to send */ 
  DMATxDescToSet = (ETH_DMADESCTypeDef*) (DMATxDescToSet->Buffer2NextDescAddr);    


  /* Return SUCCESS */
  return ETH_SUCCESS;   
}


/*******************************************************************************
* Function Name  : ETH_GetCurrentTxBuffer
* Description    : Return the address of the buffer pointed by the current descritor.
* Input          : None
* Output         : None
* Return         : Buffer address
*******************************************************************************/
u32 ETH_GetCurrentTxBuffer(void)
{ 
  /* Return Buffer address */
  return (DMATxDescToSet->Buffer1Addr);   
}


void vEMACWaitForInput( void )
{
	/* Just wait until we are signled from an ISR that data is available, or
	we simply time out. */
	xSemaphoreTake( s_xSemaphore, emacBLOCK_TIME_WAITING_FOR_INPUT );
}

static void
arp_timer(void *arg)
{
  etharp_tmr();
  sys_timeout(ARP_TMR_INTERVAL, arp_timer, NULL);
}
